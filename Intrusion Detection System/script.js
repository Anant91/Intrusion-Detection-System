function predict() {
    let input = document.getElementById("features").value;
    let features = input.split(",").map(Number);

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: features })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            data.result.includes("Intrusion")
            ? "🚨 " + data.result
            : "✅ " + data.result;
    });
}