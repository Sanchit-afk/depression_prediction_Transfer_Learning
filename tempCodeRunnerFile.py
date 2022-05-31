@app.route("/result")
def result():

    # return rendered about.html page
    return render_template("result.html")