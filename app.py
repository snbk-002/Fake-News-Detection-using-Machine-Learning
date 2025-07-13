'''
from flask import Flask, render_template, request
from utils import ml_predict, gemini_judge

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news = request.form["news"]
        local_result = ml_predict(news)
        llm_result = gemini_judge(news)

        result = {
            "news": news,
            "local_model": local_result,
            "gemini_model": llm_result
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
'''
from flask import Flask, render_template, request, redirect, url_for
from utils import ml_predict, gemini_judge, clear_chat_history

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news = request.form.get("news")
        if news:
            local_result = ml_predict(news)
            llm_result = gemini_judge(news)
            result = {
                "local_model": local_result,
                "gemini_model": llm_result
            }
    return render_template("index.html", result=result)

@app.route("/refresh", methods=["POST"])
def refresh():
    clear_chat_history()  # Clear the chat history
    return redirect(url_for("index"))  # Redirect back to the main page

if __name__ == "__main__":
    app.run(debug=True)
 