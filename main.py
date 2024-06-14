import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Vectorizer and model for task prioritization
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)

# If there are existing tasks, fit the model
if not tasks.empty:
    model.fit(tasks['description'], tasks['priority'])

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Function to add a task to the list
def add_task(description, priority):
    global tasks  
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    model.fit(tasks['description'], tasks['priority'])  # Refit model with new task
    save_tasks()

# Function to remove a task by description
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    model.fit(tasks['description'], tasks['priority'])  # Refit model without the removed task
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to recommend tasks based on a description
def recommend_task():
    if tasks.empty:
        print("No tasks available for recommendations.")
        return
    description = input("Enter task description to base recommendation on: ")
    predicted_priority = model.predict([description])[0]
    recommendations = tasks[tasks['priority'] == predicted_priority]
    if not recommendations.empty:
        print("Recommended tasks based on description similarity:")
        print(recommendations)
    else:
        print("No similar tasks found.")

# Main app
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Thanks for using....Exiting.....")
        break

    else:
        print("Invalid option. Please select a valid option.")
