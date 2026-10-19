# Employee Management System (Python + MySQL)

A command-line app for managing **employee records** stored in a MySQL database. You can add, remove, promote (give a salary raise) and list employees from a simple numbered menu.

## What it does

1. Reads the database settings (host, port, user, password, database name) from **environment variables**, so no password is written in the code.
2. Connects to MySQL and gives a clear message if the login fails or the database doesn't exist.
3. Creates the `employees` table automatically if it isn't there yet.
4. Shows a menu:

   | Choice | Action |
   |--------|--------|
   | 1 | **Add** an employee: ID, name, post and salary. Refuses duplicate IDs and invalid salaries. |
   | 2 | **Remove** an employee by ID |
   | 3 | **Promote** an employee: adds a raise to their salary and shows the new total |
   | 4 | **Display** all employees, sorted by ID |
   | 5 | Exit |

5. Closes the database connection cleanly on exit, including when you press Ctrl+C.

## Concepts covered

- **CRUD with SQL** — the four basic database operations: **C**reate (`INSERT`), **R**ead (`SELECT`), **U**pdate (`UPDATE`) and **D**elete (`DELETE`).
- **Parameterized queries** — every query passes user input separately from the SQL text (`WHERE id = %s`, with the value supplied alongside). The database driver escapes the value, so input like `1 OR 1=1` is treated as plain text and can't change the query. This is the standard defence against **SQL injection**.
- **Transactions: commit and rollback** — changes are only saved when `commit()` is called. If a query fails partway through, `rollback()` undoes it, so the database never ends up half-updated.
- **Checking whether a row exists** — `SELECT 1 ... LIMIT 1` followed by `fetchone()` reliably says whether a matching row was found. `cursor.rowcount` isn't reliable for this: after a `SELECT` on MySQL's default (unbuffered) cursor, it is `-1` until the rows have been fetched.
- **Updating in one statement** — `UPDATE employees SET salary = salary + %s` lets the database do the arithmetic. That is shorter and safer than reading the salary into Python, adding to it and writing it back, where two raises at the same moment could overwrite each other.
- **Money as `Decimal`** — salaries are stored as `DECIMAL(12, 2)` and handled with Python's `Decimal`, not `float`. Floats can't represent amounts like 0.10 exactly, which causes rounding errors in money calculations. Mixing the two (`Decimal + float`) also raises a `TypeError`.
- **Configuration from environment variables** — keeping credentials out of the source code means the code can be shared or pushed to GitHub without leaking a password, and different machines can use different settings.
- **Separating database code from user interaction** — the `insert_employee` / `delete_employee` / `raise_salary` functions only talk to the database. The menu functions only handle `input()` and `print()`. Each part can be changed or tested on its own.

## Project structure

```
main.py      # the whole app, organized into an AppConfig dataclass,
              # database functions, input helpers, menu actions and an entry point
README.md    # this file
```

- **Configuration:** `AppConfig` holds the database settings, filled in from environment variables by `AppConfig.from_env()`.
- **Database functions** (no user interaction): `connect`, `create_table`, `employee_exists`, `insert_employee`, `delete_employee`, `raise_salary`, `fetch_employees`.
- **Input helpers:** `ask_text` (asks again on empty input) and `ask_amount` (accepts only valid, non-negative amounts).
- **Menu actions:** `add_employee`, `remove_employee`, `promote_employee`, `display_employees`, tied together by `menu()`.
- **Entry point:** `run_app()` connects, creates the table if needed, runs the menu and always closes the connection.

## Requirements

- Python 3.8+
- A running MySQL server (or MariaDB)
- The MySQL connector for Python:

```bash
pip install mysql-connector-python
```

## Setup

**1. Create the database** (the app creates the table itself):

```sql
CREATE DATABASE emp;
```

**2. Set the connection details** as environment variables. Only the ones that differ from the defaults are needed:

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMP_DB_HOST` | `localhost` | MySQL server address |
| `EMP_DB_PORT` | `3306` | MySQL port |
| `EMP_DB_USER` | `root` | MySQL user |
| `EMP_DB_PASSWORD` | *(empty)* | MySQL password |
| `EMP_DB_NAME` | `emp` | Database name |

macOS / Linux:

```bash
export EMP_DB_USER=root
export EMP_DB_PASSWORD=your_password
```

Windows (PowerShell):

```powershell
$env:EMP_DB_USER = "root"
$env:EMP_DB_PASSWORD = "your_password"
```

## How to run

```bash
python main.py
```

## Sample output

```
Welcome to Employee Management Record
Press:
1 to Add Employee
2 to Remove Employee
3 to Promote Employee
4 to Display Employees
5 to Exit
Enter your Choice: 1
Enter Employee Id: E1
Enter Employee Name: Asha
Enter Employee Post: Engineer
Enter Employee Salary: 50000
Employee Added Successfully

Enter your Choice: 3
Enter Employee's Id: E1
Enter increase in Salary: 2500.50
Employee Promoted Successfully. New salary: 52,500.50

Enter your Choice: 4
Employee Id     :  E1
Employee Name   :  Asha
Employee Post   :  Engineer
Employee Salary :  52,500.50
------------------------------------
```

## A note on the design

An earlier version of this app had a subtle bug in its "does this employee exist?" check. It ran a `SELECT` and then checked `cursor.rowcount == 1`. With MySQL's default cursor, `rowcount` stays at `-1` until the results have actually been fetched, so the check always returned *False*. As a result:

- **Remove** and **Promote** always reported "Employee does not exist", even for real employees.
- **Add** never caught duplicate IDs itself. It only failed later, when the database rejected the duplicate primary key.
- The unread results left behind by the check could also trigger MySQL's "Unread result found" error on the next query.

The fix (`fetchone() is not None`) is small, but it is a good example of why it pays to check what a library function actually returns rather than assuming.

Natural next steps would be:

- searching and editing employees;
- recording promotion history in a second table;
- adding automated tests;
- moving from the command line to a small web interface (for example with Flask).

## References

- [MySQL Connector/Python Developer Guide](https://dev.mysql.com/doc/connector-python/en/)
- [MySQL Connector/Python — `cursor.rowcount`](https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-rowcount.html)
- [OWASP — SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [Python docs — `decimal` module](https://docs.python.org/3/library/decimal.html)
- [GeeksforGeeks](https://www.geeksforgeeks.org/), used as a general reference/inspiration while working on this project.

---
*This is a personal learning project for practising Python with SQL databases.*
