import mysql.connector as myql
from datetime import date
import time

mybase = myql.connect(host='localhost', user='root', password='elam2003', database='aas')
mycursor=mybase.cursor()
try:
    mycursor.execute("drop table student")
except Exception:
    None
try:
    mycursor.execute("drop table student_att")
except Exception:
    None

'''try:
    mycursor.execute("drop table class")
except Exception:
    None'''

try:
    table1 = '''create table student (s_no varchar(100),roll_number varchar(10) primary key, name varchar(32))'''
    mycursor.execute(table1)
except Exception:
    None

try:
    table2 = '''create table student_att (subject varchar(20), s_no varchar(100), roll_no varchar(10),
    month varchar(32), date date, time_in time, buffer_in time, buffer_out time, time_out time, attendance char)'''
    mycursor.execute(table2)
except Exception:
    None


try:
    table3 = '''create table class (dept varchar(32),subject varchar(32),teacher varchar(32),date date)'''
except Exception:
    None

today = date.today()
month = today.strftime("%B")
date = today.strftime("%y:%m:%d")


def english(num):
    roll = input("enter the roll number :")
    lag=0
    time_in = input("enter the in-time(if not enter:0)(hh:mm:ss) :")
    if time_in == '0':
        time_in = '0:0:0'
        buffer_in = '0:0:0'
        buffer_out = '0:0:0'
        time_out = '0:0:0'
        atten = 'a'
    else:
        buffer_count = int(input("enter the buffer time count :"))
        for i in range(buffer_count):
            if buffer_count > 0:
                buffer_out = input("enter the buffer-out time(hh:mm:ss) :")
                buffer_in = input("enter the buffer-in time(hh:mm:ss) :")
                buf_in = buffer_in.split(":")
                buf_out = buffer_out.split(":")
                lag += int(buf_in[1]) - int(buf_out[1])
        else:
            buffer_in = '0:0:0'
            buffer_out = '0:0:0'
        if lag < 10:
            atten = "p"
        else:
            atten = "a"
        time_out = input("enter the out-time(hh:mm:ss) :")
    query = '''insert into student_att(subject,s_no,roll_no,month, date, time_in,buffer_in, buffer_out,time_out,attendance)
    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
    values = ("english",num,roll,month,date, time_in, buffer_in, buffer_out, time_out, atten)
    mycursor.execute(query, values)
    mybase.commit()
    print(mycursor.rowcount,"was inserted")


def room(course, subj, teach):
    query1 = '''insert into class(dept,subject,teacher,date)
    values(%s,%s,%s,%s)'''
    values2 = (course, subj, teach, date)
    mycursor.execute(query1, values2)
    mybase.commit()


start = True
while start:
    dept = input("enter the department :")
    sub = input("enter the subject :")
    done = True
    count = 0
    if sub == "english":
        teacher = input("enter the teacher name:")
        min = int(today.strftime("%m"))
        while done:
            count += 1
            english(count)
            if count == 1:
                done=False
                room(dept,sub,teacher)
                presence=0
                count = 0

mycursor.close()
mybase.close()
