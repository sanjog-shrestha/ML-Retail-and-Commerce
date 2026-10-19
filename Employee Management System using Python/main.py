"""
Employee Management System (MySQL).

A command-line app for keeping employee records in a MySQL database: add,
remove, promote (raise salary) and list employees. Database settings come
from environment variables via an AppConfig dataclass, the table is created
automatically if it doesn't exist, and every query uses parameters so user
input can never be executed as SQL.
"""
!pip install mysql-connector-python
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

import mysql.connector
from mysql.connector import errorcode


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "emp"
    table: str = "employees"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Read settings from EMP_DB_* environment variables (falling back to defaults)."""
        return cls( 
            host=os.getenv("EMP_DB_HOST", cls.host),
            port=int(os.getenv("EMP_DB_PORT", cls.port)),
            user=os.getenv("EMP_DB_USER", cls.user),
            password=os.getenv("EMP_DB_PASSWORD", cls.password),
            database=os.getenv("EMP_DB_NAME", cls.database),
        )


# --------------------------------------------------------------------------- #
# Database layer: no input() or print() here, only SQL
# --------------------------------------------------------------------------- #
def connect(config: AppConfig):
    # First, connect to the MySQL server without selecting a database
    con = mysql.connector.connect(
        host=config.host, port=config.port, user=config.user,
        password=config.password
    )
    # Create the database if it doesn't exist
    cursor = con.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.database}")
    cursor.close()
    con.commit()
    
    # Reconnect/select the target database
    con.database = config.database
    return con


def create_table(con, config: AppConfig) -> None: 
    cursor = con.cursor()
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {config.table} (
                id       VARCHAR(20)    PRIMARY KEY,
                name     VARCHAR(100)   NOT NULL,
                position VARCHAR(100)   NOT NULL,
                salary   DECIMAL(12, 2) NOT NULL CHECK (salary >= 0)
            )"""
    )
    con.commit()
    cursor.close()


def employee_exists(con, config: AppConfig, employee_id: str) -> bool:
    cursor = con.cursor()
    cursor.execute(f"SELECT 1 FROM {config.table} WHERE id = %s LIMIT 1", (employee_id,))
    found = cursor.fetchone() is not None
    cursor.close()
    return found


def insert_employee(con, config: AppConfig, employee_id: str, name: str,
                    position: str, salary: Decimal) -> None:
    cursor = con.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {config.table} (id, name, position, salary) VALUES (%s, %s, %s, %s)",
            (employee_id, name, position, salary),
        )
        con.commit()
    except mysql.connector.Error:
        con.rollback()
        raise
    finally:
        cursor.close()


def delete_employee(con, config: AppConfig, employee_id: str) -> None:
    cursor = con.cursor()
    try:
        cursor.execute(f"DELETE FROM {config.table} WHERE id = %s", (employee_id,))
        con.commit()
    except mysql.connector.Error:
        con.rollback()
        raise
    finally:
        cursor.close()


def raise_salary(con, config: AppConfig, employee_id: str, amount: Decimal) -> Decimal:
    """Add `amount` to the employee's salary in one UPDATE and return the new salary."""
    cursor = con.cursor()
    try:
        cursor.execute(
            f"UPDATE {config.table} SET salary = salary + %s WHERE id = %s",
            (amount, employee_id),
        )
        cursor.execute(f"SELECT salary FROM {config.table} WHERE id = %s", (employee_id,))
        new_salary = cursor.fetchone()[0]
        con.commit()
        return Decimal(str(new_salary))
    except mysql.connector.Error:
        con.rollback()
        raise
    finally:
        cursor.close()


def fetch_employees(con, config: AppConfig) -> list:
    cursor = con.cursor()
    cursor.execute(f"SELECT id, name, position, salary FROM {config.table} ORDER BY id")
    rows = cursor.fetchall()
    cursor.close()
    return rows


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def ask_text(prompt: str) -> str:
    """Keep asking until the user types something non-empty."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This field can't be empty.")


def ask_amount(prompt: str) -> Optional[Decimal]:
    """Read a non-negative money amount; return None if the input isn't valid."""
    raw = input(prompt).strip()
    try:
        amount = Decimal(raw).quantize(Decimal("0.01"))
    except InvalidOperation:
        print(f"'{raw}' is not a valid amount.")
        return None
    if amount < 0:
        print("The amount can't be negative.")
        return None
    return amount


# --------------------------------------------------------------------------- #
# Menu actions: input() and print() live here
# --------------------------------------------------------------------------- #
def add_employee(con, config: AppConfig) -> None:
    employee_id = ask_text("Enter Employee Id: ")
    if employee_exists(con, config, employee_id):
        print("Employee already exists. Please try again.")
        return
    name = ask_text("Enter Employee Name: ")
    position = ask_text("Enter Employee Post: ")
    salary = ask_amount("Enter Employee Salary: ")
    if salary is None:
        return
    try:
        insert_employee(con, config, employee_id, name, position, salary)
        print("Employee Added Successfully")
    except mysql.connector.Error as err:
        print(f"Error: {err}")


def remove_employee(con, config: AppConfig) -> None:
    employee_id = ask_text("Enter Employee Id: ")
    if not employee_exists(con, config, employee_id):
        print("Employee does not exist. Please try again.")
        return
    try:
        delete_employee(con, config, employee_id)
        print("Employee Removed Successfully")
    except mysql.connector.Error as err:
        print(f"Error: {err}")


def promote_employee(con, config: AppConfig) -> None:
    employee_id = ask_text("Enter Employee's Id: ")
    if not employee_exists(con, config, employee_id):
        print("Employee does not exist. Please try again.")
        return
    amount = ask_amount("Enter increase in Salary: ")
    if amount is None:
        return
    try:
        new_salary = raise_salary(con, config, employee_id, amount)
        print(f"Employee Promoted Successfully. New salary: {new_salary:,.2f}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")


def display_employees(con, config: AppConfig) -> None:
    try:
        employees = fetch_employees(con, config)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return
    if not employees:
        print("No employees found.")
        return
    for emp_id, name, position, salary in employees:
        print("Employee Id     : ", emp_id)
        print("Employee Name   : ", name)
        print("Employee Post   : ", position)
        print(f"Employee Salary :  {Decimal(str(salary)):,.2f}")
        print("------------------------------------")


MENU = {
    "1": ("Add Employee", add_employee),
    "2": ("Remove Employee", remove_employee),
    "3": ("Promote Employee", promote_employee),
    "4": ("Display Employees", display_employees),
}


def menu(con, config: AppConfig) -> None:
    while True:
        print("\nWelcome to Employee Management Record")
        print("Press:")
        for key, (label, _) in MENU.items():
            print(f"{key} to {label}")
        print("5 to Exit")

        choice = input("Enter your Choice: ").strip()
        if choice == "5":
            print("Exiting the program. Goodbye!")
            break
        if choice in MENU:
            MENU[choice][1](con, config)
        else:
            print("Invalid Choice! Please try again.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_app(config: Optional[AppConfig] = None) -> None:
    config = config or AppConfig.from_env()
    try:
        con = connect(config)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Could not log in to MySQL: check EMP_DB_USER / EMP_DB_PASSWORD.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Database '{config.database}' does not exist. Create it with: "
                  f"CREATE DATABASE {config.database};")
        else:
            print(f"Could not connect to MySQL: {err}")
        return

    try:
        create_table(con, config)
        menu(con, config)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting the program. Goodbye!")
    finally:
        con.close()


if __name__ == "__main__":
    run_app()