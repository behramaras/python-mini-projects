# Book Registration Automation

A simple **Python Tkinter** desktop application for managing book records. It allows you to add, list, edit, and delete books with a user-friendly interface. The data is stored using **SQLite**.

## Features

* Add new books
* List all books
* Search books by title, author, or publisher
* Edit book information
* Delete books from the database
* Copy row data to clipboard
* Refresh the table

## Requirements

* Python 3.x
* Tkinter (comes with Python)
* SQLite3 (comes with Python)

## Project Structure

```
python-mini-projects/
└── book-registration-automation/
    ├── app.py                  # Main application file
    ├── baglan.sql              # SQLite database file (created automatically)
    └── README.md               # Project documentation
```

> Note: `app.py` is the main program file. Running this will launch the GUI.

## Installation & Running

1. Make sure Python 3 is installed.
2. Navigate to the project directory:

```bash
cd python-mini-projects/book-registration-automation
```

3. Run the application:

```bash
python app.py
```

The program will create the database file `baglan.sql` automatically if it doesn’t exist.

## Usage

1. **Main Screen Buttons**:

   * **Add Book**: Open the form to add a new book.
   * **List Books**: Show all books in a table with options to edit, delete, search, and copy data.

2. **Add Book Form**: Fill out the following fields:

   * Book Name
   * Author
   * Publisher
   * Number of Pages
   * Edition
   * Publication Year

3. **Book List Table**:

   * Use the search bar to find books.
   * Right-click on a row to edit or copy the row.
   * Edit window allows you to update or delete records.

4. Developer info and GitHub link appear at the bottom of the app.

## Database

* SQLite database: `baglan.sql`
* Table: `kitaplar`
* Columns:

  * `id` (INTEGER PRIMARY KEY)
  * `Kitap_Adi` (VARCHAR(45))
  * `Yazar` (VARCHAR(45))
  * `Yayin_Evi` (VARCHAR(45))
  * `Sayfa_Sayisi` (INT)
  * `Baski` (INT)
  * `Yayin_Yili` (INT)

## Developer

**Behram Aras**
[GitHub Profile](https://github.com/behramaras)
