import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
import pandas as pd
import mysql.connector
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual Gemini API key
genai.configure(api_key="AIzaSyADe2YtszUhEcTR546CNlc3UVssE7PStMU")

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === MySQL Database Configuration ===
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "srikanth123",
    "database": "store"
}

# Keep track of uploaded tables
uploaded_tables = set()

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_database_tables():
    """Retrieves a list of tables from the MySQL database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        return set(tables) # Return as a set for easier union
    except mysql.connector.Error as err:
        logging.error(f"Error fetching database tables: {err}")
        return set()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- Helper: Get DB schema for a specific table ---
def get_schema(table_name="data"):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"DESCRIBE `{table_name}`")
    except mysql.connector.Error as e:
        cur.close()
        conn.close()
        raise e
    schema = [{"name": row[0], "type": row[1]} for row in cur.fetchall()]
    cur.close()
    conn.close()
    logging.debug(f"Schema for table '{table_name}': {schema}")
    return schema

# --- Helper: Run SQL query ---
def run_sql(query):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

# --- Helper: Generate SQL with Gemini for specific table ---
# --- Helper: Generate SQL with Gemini for specific table ---
def generate_sql(prompt, schema, table_name="data"):
    logging.debug(f"Generating SQL for prompt: '{prompt}' with schema: {schema} and table: '{table_name}'")
    system_prompt = "You are an assistant that converts natural language questions into valid MySQL queries.\n"
    system_prompt += f"Database schema:\nTable: `{table_name}`\nColumns:\n" # Enclose table name in backticks
    for col in schema:
        system_prompt += f"- `{col['name']}` ({col['type']})\n" # Enclose column name in backticks
    user_prompt = f"{system_prompt}\nUser question: {prompt}\nGenerate ONLY the MySQL query. Ensure all table and column names are enclosed in backticks." # Explicit instruction

    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(user_prompt)
        sql = response.text.strip()

        # Clean code fences and leading 'sql' if present
        if sql.startswith("```"):
            # Remove triple backticks and language tag if exists
            sql = sql.strip("```").strip()
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()

        # Also remove any leading 'sql' if present without backticks
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

        logging.debug(f"Generated SQL query: {sql}")
        return sql
    except Exception as e:
        logging.error(f"Error during Gemini SQL generation: {e}")
        raise e

# === Routes ===
@app.route("/")
def index():
    database_tables = get_database_tables()
    return render_template("index.html", database_tables=database_tables, uploaded_tables=list(uploaded_tables))

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Get the optional table name from the form data
    table_name = request.form.get("tablename", "").strip()
    if not table_name:
        # Use the file name (without extension and spaces replaced) if no table name provided
        table_name = file.filename.rsplit('.', 1)[0].replace(" ", "_")

    # Validate table_name: allow only letters, numbers, underscore (you can improve this regex)
    import re
    if not re.match(r'^[A-Za-z0-9_]+$', table_name):
        return "Invalid table name. Use only letters, numbers, and underscores.", 400

    # Read file into pandas dataframe
    try:
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        elif file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="ISO-8859-1")
        else:
            return "Unsupported file format. Please upload .csv or .xlsx", 400
    except Exception as e:
        return f"Failed to read file: {e}", 400

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Drop old table if exists
        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")

        # Create table: all columns as TEXT type for simplicity
        create_sql = f"CREATE TABLE `{table_name}` ("
        for col in df.columns:
            col_clean = col.replace("`", "")  # sanitize column name by removing backticks if any
            create_sql += f"`{col_clean}` TEXT, "
        create_sql = create_sql.rstrip(", ") + ")"
        cur.execute(create_sql)

        # Insert data row by row
        for _, row in df.iterrows():
            placeholders = ", ".join(["%s"] * len(row))
            columns = ", ".join(f"`{col.replace('`', '')}`" for col in df.columns)
            insert_sql = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
            cur.execute(insert_sql, tuple(row))
        conn.commit()
        uploaded_tables.add(table_name) # Add the uploaded table to the set
    except Exception as e:
        return f"Failed to upload data to MySQL: {e}", 500
    finally:
        cur.close()
        conn.close()

    return f"File uploaded and converted to MySQL table '{table_name}' successfully.", 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("message", "").strip()
    table_name = data.get("table", "data")  # Default table "data"

    all_tables = get_database_tables().union(uploaded_tables)
    if table_name not in all_tables:
        return jsonify({"reply": f"Table '{table_name}' not found. Available tables: {list(all_tables)}"})

    if not user_prompt:
        return jsonify({"reply": "Please send a valid message."})

    try:
        schema = get_schema(table_name) # Fetch the schema for the specified table
        sql = generate_sql(user_prompt, schema, table_name)
    except Exception as e:
        return jsonify({"reply": f"Failed to generate SQL: {e}"})

    result = run_sql(sql)

    if "error" in result:
        return jsonify({
            "reply": f"Error executing SQL:\n{result['error']}",
            "sql": sql
        })
    elif result and "rows" in result and "columns" in result:
        # Format the result into a dictionary for cleaner output
        output = {}
        if result["columns"] and len(result["rows"]) > 0:
            if len(result["columns"]) == 1:
                output[result["columns"][0]] = [row[0] for row in result["rows"]]
            else:
                for i, column in enumerate(result["columns"]):
                    output[column] = [row[i] for row in result["rows"]]
        elif result["columns"]:
            output["columns"] = result["columns"]
            output["rows"] = []
        else:
            output["message"] = "No data found."
        return jsonify(output)
    else:
        return jsonify({"reply": "No result or an unexpected result format."})
    
@app.route("/tables", methods=["GET"])
def list_tables():
    database_tables = get_database_tables()
    return jsonify({"database_tables": list(database_tables), "uploaded_tables": list(uploaded_tables)})

if __name__ == "__main__":
    app.run(debug=True)

"""import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
import pandas as pd
import mysql.connector
import os

# === Gemini API Key ===
genai.configure(api_key="AIzaSyADe2YtszUhEcTR546CNlc3UVssE7PStMU")  # Replace with your actual key

app = Flask(__name__)

# === MySQL DB Config ===
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "srikanth123",
    "database": "store"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ✅ Get Full Schema for Entire MySQL Database
def get_full_schema():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [row[0] for row in cur.fetchall()]

    schema = {}
    for table in tables:
        cur.execute(f"DESCRIBE `{table}`")
        columns = [{"name": row[0], "type": row[1]} for row in cur.fetchall()]
        schema[table] = columns

    cur.close()
    conn.close()
    return schema

# ✅ Run SQL
def run_sql(query):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

# ✅ Generate SQL from Gemini using full DB schema
def generate_sql(prompt, schema):
    system_prompt = "You are an assistant that converts natural language to MySQL queries.\n"
    system_prompt += "Database Schema:\n"
    for table, columns in schema.items():
        system_prompt += f"Table: {table}\n"
        for col in columns:
            system_prompt += f" - {col['name']} ({col['type']})\n"
    system_prompt += f"\nUser Question: {prompt}\nGenerate ONLY the MySQL query."

    model = genai.GenerativeModel("gemini-2.0-flash")  # You can also use "gemini-2.0-flash"
    response = model.generate_content(system_prompt)

    sql = response.text.strip()
    if sql.startswith("```"):
        sql = sql.strip("```").replace("mysql", "").strip()

    return sql

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_prompt = request.json.get("message", "").strip()
    if not user_prompt:
        return jsonify({"reply": "Please send a valid message."})

    schema = get_full_schema()
    sql = generate_sql(user_prompt, schema)
    result = run_sql(sql)

    if "error" in result:
        return jsonify({"reply": f"SQL Error:\n{result['error']}", "sql": sql})

    return jsonify({"reply": result, "sql": sql})

if __name__ == "__main__":
    app.run(debug=True)
"""





"""import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
import pandas as pd
import sqlite3
import os
import openai

genai.configure(api_key="AIzaSyADe2YtszUhEcTR546CNlc3UVssE7PStMU")

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_data"
DB_PATH = os.path.join(UPLOAD_FOLDER, "data.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# --- Helper: Get DB schema ---
def get_schema():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(data)")
    schema = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]
    conn.close()
    return schema

# --- Helper: Run SQL query ---
def run_sql(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        conn.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

# --- Helper: Generate SQL with OpenAI ---

# Set Gemini API key

def generate_sql(prompt, schema):
    system_prompt = "You are an assistant that converts natural language questions into valid SQLite SQL queries.\n"
    system_prompt += "Database schema:\nTable: data\nColumns:\n"
    for col in schema:
        system_prompt += f"- {col['name']} ({col['type']})\n"

    user_prompt = f"{system_prompt}\nUser question: {prompt}\nGenerate ONLY the SQLite query."

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(user_prompt)

    sql = response.text.strip()

    # 🧼 Clean unwanted formatting
    if sql.startswith("```"):
        sql = sql.strip("```")  # remove all backticks
        sql = sql.replace("sqlite", "").strip()  # remove language tag if present

    return sql



# === Routes ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    print(file.filename)
    if file.filename == "":
        return "No selected file"
    if file:
        try:
            if file.filename.endswith(".xlsx"):
                df = pd.read_excel(file, engine="openpyxl")
            elif file.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding="ISO-8859-1")
            else:
                return "Unsupported file format. Please upload .csv or .xlsx"
        except Exception as e:
            return f"Failed to read file: {e}"
        name=file.filename
        conn = mysql.connect(DB_PATH)
        files=df.to_sql(name, conn, ,index=False)
        print(files)
        conn.close()
        return "File uploaded and converted to database successfully."

@app.route("/chat", methods=["POST"])
def chat():
    user_prompt = request.json.get("message", "").strip()
    if not user_prompt:
        return jsonify({"reply": "Please send a valid message."})

    schema = get_schema()
    sql = generate_sql(user_prompt, schema)

    result = run_sql(sql)
    if "error" in result:
        return jsonify({
            "reply": f"Error executing SQL:\n{result['error']}",
            "sql": sql
        })

    return jsonify({
        "reply": result,
        "sql": sql
    })

if __name__ == "__main__":
    app.run(debug=True)
"""