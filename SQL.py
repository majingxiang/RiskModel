import pandas as pd
import pymysql
from sqlalchemy import *
import MySQLdb
from loguru import logger


def connect_to_sql(database="prod", username="root", password="1222"):
    try:
        # Notice: this charset is necessary, otherwise Chinese character cannot displayed
        # This autocommit is very very important
        connection = MySQLdb.connect(host='localhost', user=username, password=password, db=database, charset='utf8',
                                     autocommit=True)
        logger.info("Connected to MySQL {}".format(database))
        cursor = connection.cursor()
        engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format(username, password, database))
        return connection, cursor, engine
    except pymysql.Error as msg:
        exit(logger.warning(msg))


# mutual fund nav and holding
# todo: finish this part in the future

# factor data and stock price (derive the return)
start_date, end_date = '20140101', '20200101'

fundamental_query = "select * from prod.equity_valuation where date between '{}' and '{}'".format(start_date, end_date)

sector_query = "select date, name as industry, close as sector_close from prod.sector_daily_price where date " \
               "between '{}' and '{}'".format(start_date, end_date)

price_query = "select p.date, p.ticker, m.sw_l1_name as industry, p.close from prod.equity_daily_price p join " \
              "prod.equity_map m on p.ticker = m.ticker where p.date between '{}' and '{}' and" \
              " p.ticker not like '%%ST%%' ".format(start_date, end_date)

if __name__ == "__main__":
    connection, cursor, engine = connect_to_sql()

    fundamental_factor = pd.read_sql(fundamental_query, con=connection)
    sector = pd.read_sql(sector_query, con=connection)
    price = pd.read_sql(price_query, con=connection)

    price.to_csv("price.csv", index=False)
    sector.to_csv("sector.csv", index=False)
    fundamental_factor.to_csv("fundamental.csv", index=False)
