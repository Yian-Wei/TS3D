import re
import sqlparse
import networkx as nx
import matplotlib.pyplot as plt
from sqlparse.sql import Identifier, IdentifierList, Token, Parenthesis, Function
from sqlparse.tokens import Keyword, DML, Punctuation, Whitespace

def extract_subqueries(sql):
    """
    从主查询中提取所有子查询。
    :param sql: 输入的 SQL 查询字符串
    :return: 一个包含子查询的列表
    """
    def find_subqueries(parsed):
        subqueries = []
        for token in parsed.tokens:
            if isinstance(token, sqlparse.sql.Parenthesis):  # 检查括号内容是否为子查询
                subquery = token.value[1:-1].strip()  # 去掉括号
                subqueries.append(subquery)
                # 递归检查嵌套子查询
                # print(sqlparse.parse(subquery)[0])
                # subqueries.extend(find_subqueries(sqlparse.parse(subquery)[0]))
            elif token.is_group:  # 继续解析分组结构
                subqueries.extend(find_subqueries(token))
        return subqueries
    # 对 SQL 进行解析
    parsed = sqlparse.parse(sql)[0]
    return find_subqueries(parsed)


def extract_set_clause(update_sql):
    """
    提取 UPDATE 语句中的 SET 子句的字段名和对应的值。
    :param update_sql: 输入的 UPDATE SQL 语句
    :return: 包含字段和值的列表
    """
    # 移除子查询部分
    update_sql = remove_subqueries(update_sql)

    parsed = sqlparse.parse(update_sql)[0]  # 解析 SQL 语句
    tokens = parsed.tokens

    # 查找 SET 子句
    set_clause = None
    flag = False
    for token in tokens:

        if flag and token.ttype is not Whitespace:
            set_clause = token.value
            break
        if token.ttype is Keyword and token.value.upper().startswith("SET"):
            flag = True

    # 提取字段和值
    if set_clause:
        fields_values = [
            item.strip().split("=") for item in set_clause.split(",")
        ]

        # return [{"field": fv[0].strip(), "value": fv[1].strip()} for fv in fields_values]
        return [fv[0].strip() for fv in fields_values]

    return []



def remove_subqueries(sql):
    """
    使用正则表达式移除子查询部分。
    支持处理嵌套括号和子查询中的逗号。
    """
    stack = []
    result = []
    i = 0

    while i < len(sql):
        char = sql[i]

        # 检测子查询的开始
        if char == '(' and sql[i + 1:i + 7].upper() == 'SELECT':
            stack.append('(')  # 子查询开始入栈
            i += 1
            while stack and i < len(sql):
                # 检测括号的匹配
                if sql[i] == '(':
                    stack.append('(')
                elif sql[i] == ')':
                    stack.pop()
                i += 1
        else:
            result.append(char)
            i += 1

    return ''.join(result)




def extract_tables_from_sql(sql_query):
    """
    从SQL查询中提取所有涉及的表名（包括FROM子句中的表和子查询）。
    """
    parsed = sqlparse.parse(sql_query)
    tables = set()

    for stmt in parsed:
        from_seen = False  # 是否已找到 FROM 关键字
        for token in stmt.tokens:
            # 忽略空白字符或标点符号
            if token.ttype in (Whitespace, Punctuation):
                continue

            # 找到 FROM 关键字
            if from_seen == True:
                # 如果是标识符列表（如多个表）
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        tables.add(identifier.get_real_name())
                # 如果是单个标识符
                elif isinstance(token, Identifier):
                    tables.add(token.get_real_name())
                # 子查询的别名
                elif token.ttype == Keyword and token.value.upper() == 'SELECT':
                    # 子查询可能会嵌套表名
                    subquery_tables = extract_tables_from_sql(str(token))
                    tables.update(subquery_tables)

                # 退出，避免处理额外部分
                from_seen = False

                # 标记FROM出现
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True

    return tables

def extract_fields(token):
    fields = []
    if isinstance(token, IdentifierList):
        for item in token.get_identifiers():
            if isinstance(item, Identifier):
                fields.append(item.get_real_name())
            if isinstance(item, Function):
                fields.append(item.tokens[1].value.strip('()'))

    elif isinstance(token, Identifier):
        fields.append(token.get_real_name())

    elif isinstance(token, Function):
        fields.append(token.tokens[1].value.strip('()'))



    return fields

