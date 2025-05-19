function askBot() {
    const msg = document.getElementById("question").value;
    const table = document.getElementById("table").value || "data"; // default "data"

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ 
            message: msg,
            table: table
        })
    })
    .then(res => res.json())
    .then(data => {
        const resBox = document.getElementById("response");
        if (typeof data.reply === 'string') {
            resBox.textContent = data.reply;
        } else if (data.reply.columns && data.reply.rows) {
            let text = data.reply.columns.join(" | ") + "\n" + "-".repeat(40) + "\n";
            data.reply.rows.forEach(row => {
                text += row.join(" | ") + "\n";
            });
            resBox.textContent = text;
        } else {
            resBox.textContent = JSON.stringify(data.reply);
        }
    })
    .catch(err => {
        document.getElementById("response").textContent = "Error: " + err;
    });
}
