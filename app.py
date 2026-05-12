from flask import Flask, render_template, request, redirect, url_for, flash
import DBManager
from recommender import KNN
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET")

@app.route("/", methods=["GET", "POST"])
def index():
    user_id = request.args.get("user_id")
    history = []

    if user_id:
        try:
            user_id = int(user_id)
            history = DBManager.get_user_sessions(user_id)
        except ValueError:
            flash("Invalid User ID")
            return redirect(url_for("index"))
    
    return render_template("index.html", user_id=user_id, history=history)

@app.route("/input", methods=["GET", "POST"])
def input_data():
    user_id = request.args.get("user_id")
    
    if not user_id:
        return redirect(url_for("index"))
    
    cannabinoids = DBManager.get_all_cannabinoids()
    terpenes = DBManager.get_all_terpenes()
    tags = DBManager.get_all_tags()
    
    predicted_effects = []
    
    if request.method == "POST":
        mode = request.form.get("mode")
        
        if mode == "add_tag":
            new_tag = request.form.get("new_tag_name")
            if new_tag:
                try:
                    DBManager.add_tag(new_tag)
                    flash(f"New effect '{new_tag}' added!")
                except Exception as e:
                    flash(f"Error adding effect: {e}")
            return redirect(url_for("input_data", user_id=user_id))
        
        if mode == "add_terpene":
            new_terpene = request.form.get("new_terpene_name")
            if new_terpene:
                try:
                    DBManager.add_terpene(new_terpene)
                    flash(f"New terpene '{new_terpene}' added!")
                except Exception as e:
                    flash(f"Error adding terpene: {e}")
            return redirect(url_for("input_data", user_id=user_id))

        if mode == "add_cannabinoid":
            new_cannabinoid = request.form.get("new_cannabinoid_name")
            if new_cannabinoid:
                try:
                    DBManager.add_cannabinoid(new_cannabinoid)
                    flash(f"New cannabinoid '{new_cannabinoid}' added!")
                except Exception as e:
                    flash(f"Error adding cannabinoid: {e}")
            return redirect(url_for("input_data", user_id=user_id))

        name = request.form.get("name")
        brand = request.form.get("brand")
        product_type = request.form.get("type")
        notes = request.form.get("notes")
        servings = float(request.form.get("servings", 1.0))
        
        #Parse cannabinoids
        selected_cannabinoids = []
        for c in cannabinoids:
            val = request.form.get(f"cannabinoid_{c['id']}")

            if val and float(val) > 0:
                selected_cannabinoids.append({"id": c['id'], "serving": float(val)})
        
        #Parse terpenes
        selected_terpenes = []
        for t in terpenes:
            if request.form.get(f"terpene_{t['id']}"):
                selected_terpenes.append(t['id'])
        
        mode = request.form.get("mode") #'history' or 'query'
        
        if mode == "history":
            selected_tags = []

            for tag in tags:
                if request.form.get(f"tag_{tag['id']}"):
                    selected_tags.append(tag['id'])
            
            #Save to DB
            p_id = DBManager.add_product(name, brand, product_type, notes, selected_cannabinoids, selected_terpenes)
            DBManager.add_session(user_id, p_id, servings, notes, selected_tags)
            flash("History recorded successfully!")

            return redirect(url_for("input_data", user_id=user_id))
        elif mode == "query":
            #Predict effects
            knn = KNN(int(user_id))

            cannabinoids = []
            for c in selected_cannabinoids:
                cannabinoids.append({
                    'cannabinoid_id': c['id'],
                    'serving': c['serving']
                })

            terpenes = []
            for t_id in selected_terpenes:
                terpenes.append({
                    'terpene_id': t_id
                })

            query_product = {
                'cannabinoids': cannabinoids,
                'terpenes': terpenes,
                'type': product_type}
            
            predicted_effects = knn.predict(query_product, servings)

    return render_template("input.html", user_id=user_id, cannabinoids=cannabinoids, terpenes=terpenes, tags=tags, predicted_effects=predicted_effects)

if __name__ == "__main__":
    app.run(debug=True)