# Student Attendance System

This project is a **student attendance system** built with Python and MySQL. Teachers can record students' in-time, out-time, buffer times, and attendance status.

---

## Features

- Record student attendance (present / absent)
- Track in-time, out-time, and buffer time
- Automatically insert data into MySQL tables
- Department and subject-based class registration
- Full MySQL database integration

---

## Requirements

- Python 3.x
- MySQL
- Python packages:
  ```bash
  pip install mysql-connector-python

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   cd REPO_NAME

2. Create the database in MySQL:

   ```sql
   CREATE DATABASE aas;
   ```

3. Update the MySQL connection details in `attendance.py`:

   ```python
   mybase = myql.connect(
       host='localhost',
       user='root',
       password='YOUR_PASSWORD',
       database='aas'
   )
   ```

4. Run the application:

   ```bash
   python attendance.py
   ```

---

## Usage

* Enter the department and subject when prompted.
* Provide the teacher name for the selected subject.
* Enter student roll numbers, in-time, out-time, and buffer times.
* Attendance status is calculated automatically and stored in the database.

---

## Notes

* Do not hardcode database passwords in public repositories.
* Use a `.env` file and add it to `.gitignore` for security.
* Time values are stored using MySQL `TIME` format.
* Attendance is marked based on buffer time calculation.
