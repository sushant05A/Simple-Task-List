import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class TaskManagerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Task Management App")

        self.tasks = pd.DataFrame(columns=['description', 'priority'])
        self.load_tasks()

        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()
        self.model = make_pipeline(self.vectorizer, self.clf)

        if not self.tasks.empty:
            self.model.fit(self.tasks['description'], self.tasks['priority'])

        self.create_widgets()

    def load_tasks(self):
        try:
            self.tasks = pd.read_csv('tasks.csv')
        except FileNotFoundError:
            pass

    def save_tasks(self):
        self.tasks.to_csv('tasks.csv', index=False)

    def create_widgets(self):
        self.description_label = ttk.Label(self.master, text="Task Description:")
        self.description_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.description_entry = ttk.Entry(self.master)
        self.description_entry.grid(row=0, column=1, padx=5, pady=5)

        self.priority_label = ttk.Label(self.master, text="Task Priority:")
        self.priority_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")

        self.priority_entry = ttk.Combobox(self.master, values=["Low", "Medium", "High"])
        self.priority_entry.grid(row=1, column=1, padx=5, pady=5)
        self.priority_entry.current(0)

        self.add_button = ttk.Button(self.master, text="Add Task", command=self.add_task)
        self.add_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        self.remove_button = ttk.Button(self.master, text="Remove Task", command=self.remove_task)
        self.remove_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        self.list_button = ttk.Button(self.master, text="List Tasks", command=self.list_tasks)
        self.list_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="we")

        self.recommend_button = ttk.Button(self.master, text="Recommend Task", command=self.recommend_task)
        self.recommend_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="we")

    def add_task(self):
        description = self.description_entry.get()
        priority = self.priority_entry.get()
        if description.strip() == "":
            messagebox.showerror("Error", "Please enter a task description.")
            return
        new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
        self.tasks = self.tasks.append(new_task, ignore_index=True)
        self.save_tasks()
        messagebox.showinfo("Success", "Task added successfully.")
        self.description_entry.delete(0, tk.END)
        self.update_model()

    def remove_task(self):
        description = self.description_entry.get()
        if description.strip() == "":
            messagebox.showerror("Error", "Please enter a task description to remove.")
            return
        self.tasks = self.tasks[self.tasks['description'] != description]
        self.save_tasks()
        messagebox.showinfo("Success", "Task removed successfully.")
        self.description_entry.delete(0, tk.END)
        self.update_model()

    def update_model(self):
        if not self.tasks.empty:
            self.model.fit(self.tasks['description'], self.tasks['priority'])

    def list_tasks(self):
        if self.tasks.empty:
            messagebox.showinfo("Info", "No tasks available.")
        else:
            list_window = tk.Toplevel(self.master)
            list_window.title("Task List")
            list_text = tk.Text(list_window)
            list_text.insert(tk.END, self.tasks.to_string(index=False))
            list_text.pack()

    def recommend_task(self):
        if not self.tasks.empty:
            predictions = self.model.predict_proba(self.tasks['description'])
            highest_priority_task_index = predictions[:, 2].argmax()
            recommended_task = self.tasks.loc[highest_priority_task_index, 'description']
            messagebox.showinfo("Recommended Task", f"Recommended task: {recommended_task} - Priority: High")
        else:
            messagebox.showinfo("Info", "No tasks available for recommendations.")

def main():
    root = tk.Tk()
    app = TaskManagerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
