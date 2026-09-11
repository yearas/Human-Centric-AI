# Human-Centric AI

Semester course project for Human-Centric Artificial Intelligence, by Yunus Emre Aras and Ozan Ermis. The home page links to Projects 1, 2, 3 and 4.

## Run locally

Use Python 3.13 or newer and Git. 

### 1. Get the repository

```bash
git clone https://github.com/yearas/Human-Centric-AI.git
cd Human-Centric-AI
```

### 2. Create an environment

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Windows (Command Prompt):

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install and start Django

Run these commands from the folder containing `manage.py`:

```bash
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 8000
```

Open **[http://127.0.0.1:8000/home/](http://127.0.0.1:8000/home/)** in your browser. It redirects to the homepage. No login is needed.


## Try the projects

### Project 1: Automated Machine Learning

1. Click **Project 1**, then **Upload the csv file**. Choose a CSV and click **Upload**.
2. Choose the **X-axis** and **Y-axis**, then click **Update plot**.
3. Click **Train a model**. Choose Ridge, Decision Tree, or k-Nearest Neighbors, and select Train-Test Split or Cross-Validation.
4. Enter the hyperparameter range and click **Train**. For a quick first try, use Decision Tree, Train-Test Split, a test fraction of `0.2`, and a range from `1` to `5`.
5. Check the results table, then use **Return to Visualization** or **Return to Homepage**.

The CSV needs column headers, at least two numeric feature columns, and the target in the last column. Use data without missing values or an ID column.

Need a sample? Run this in the activated environment to create `iris.csv`, then upload it. If the server is running in that terminal, stop it first and restart it afterward.

```bash
python -c "from sklearn.datasets import load_iris; load_iris(as_frame=True).frame.to_csv('iris.csv', index=False)"
```

### Project 2: Explainability

1. Click **Project 2**. The Palmer Penguins dataset loads automatically.
2. Switch between **Decision Tree** and **Logistic Regression**, and move the **lambda** slider to compare accuracy and model complexity.
3. Change **Example x** and **Target label** to view counterfactual examples.
4. Change **Feature** to update the Partial Dependence and Accumulated Local Effects plots. Click a plot or the decision tree to open it at full size.
5. Click **Return to Homepage** when finished. The controls update the page automatically.

### Project 3: Active Learning for Learning-to-Defer

1. Click **Project 3**. The first visit downloads AG News and runs the experiments, so allow time for the page to load. Later visits use saved results.
2. Scroll through Tasks 1–4: the AI baseline, simulated expert, learning-to-defer results, and random versus active expert querying.
3. Click **Download the report** for the PDF with the experiments and results. You can also open [the report directly in this repository](project3/Report.pdf).
4. Click **Return to Homepage** when finished.


### Project 4: Preference Elicitation

1. Click **Project 4**. The landing page describes the movie preference study.
2. Click **Download the report (PDF)** for the feature representation, the ranking model and the study design. You can also open [the report directly in this repository](project4/static/project4/report.pdf).
3. Tick the consent box and click **Start the study**. You are randomly assigned to one of two interfaces: pairwise comparisons (20 questions with two movies each) or rankings (5 questions with ten movies each).
4. Pairwise: click the movie you would rather watch. Ranking: click the movies in your order of preference, starting with your favorite. The last movie is placed automatically.
5. After the last question, the result page shows ten recommended movies you have not seen during the study and how certain the model is about your taste.
6. To try the other interface, start the study again from the Project 4 page. The interface is assigned at random on each start, so it may take a few tries. Click **Return to Homepage** when finished.

The movies come from the IMDB 5000 dataset, which is included in the repository. Answers are stored in the local SQLite database created by `migrate`.
