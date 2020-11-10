import pymssql

sql = """SELECT * FROM ML.dbo.FeedbackValue"""
# default values
n, m, k, s = 5, 20, 50, 2
try:
    # server, user, password, database
    conn = pymssql.connect(server='10.100.100.114',
                           user='sa',
                           password='Password01!',
                           database='ML')
    cursor = conn.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()
    print(row)
except Exception as e:
    print(e)
