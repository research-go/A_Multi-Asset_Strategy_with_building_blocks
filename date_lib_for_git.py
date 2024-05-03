import datetime as dt


def what_date_of_months(date_list, date, kind='end'):  # input: 'yyyy-mm' or 'yyyy-mm-dd 00:00:00'. output: 'yyyy-mm-dd 00:00:00'
    """
    if input date is not business days, output will be the business day before.
    :param date_list: where you catch
    :param date: input date
    :param kind: 'end' meaning the end of the month and others are others.
    :return: the output date
    """
    def searching(searching_year, searching_month, date_kind):
        if date_kind == "end":
            target_list = list(filter(lambda x: x.year == searching_year and x.month == searching_month, date_list))
        elif type(date_kind) is int:
            target_list = list(filter(lambda x: x.year == searching_year and x.month == searching_month and x.day <= date_kind, date_list))
        else:
            raise Exception("error: what_date_of_months")
        return max(target_list)

    if type(date) == str and 2 <= len(date.split("-")) <= 3:  # 'yyyy-mm'
        p = int(date.split("-")[0])
        q = int(date.split("-")[1])
        result = searching(p, q, kind)
    elif str(type(date)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":  # 'yyyy-mm-dd 00:00:00'
        result = searching(date.year, date.month, kind)
    else:
        raise Exception("error: what_date_of_months")
    return result


def the_number_date(date, number: int) -> str:  # From ('yyyy-mm' or pandas.timestamp) to 'yyyy-mm'
    """
    what the date is when some months pass
    :param date: input date
    :param number: months pass
    :return: output date
    """
    if str(type(date)) == "<class 'str'>" and 2 <= len(date.split("-")) <= 3:
        year = int(date.split("-")[0])
        month = int(date.split("-")[1])
    elif str(type(date)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
        year = date.year
        month = date.month
    else:
        raise Exception("error: the_number_date")
    new_year = year + (month + number - 1) // 12
    new_month = (month + number - 1) % 12 + 1
    new_date = str(new_year) + "-" + str(new_month)
    return new_date


if __name__ == "__main__":
    d1 = dt.datetime.strptime("2024-04-26", "%Y-%m-%d")
    d2 = dt.datetime.strptime("2024-04-29", "%Y-%m-%d")
    d3 = dt.datetime.strptime("2024-04-30", "%Y-%m-%d")
    print(what_date_of_months([d1, d2, d3], "2024-04", kind=28))
    print(the_number_date("2018-06", 8))
