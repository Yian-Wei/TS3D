import sqlparse
import networkx as nx
from matplotlib import pyplot as plt
from sqlparse.sql import Identifier, IdentifierList, Function
from sqlparse.tokens import DML, Punctuation, Whitespace
from utils import parse


def extract_table_field_dependencies(query, dependencies):
    # 判断是否有子查询
    subqueries = parse.extract_subqueries(query)

    subqueries.append(query)

    for i, query in enumerate(subqueries):
        # 使用sqlparse库解析SQL查询
        parsed = sqlparse.parse(query)
        for stmt in parsed:
            operation = None
            table = None
            fields = []

            for token in stmt.tokens:
                if token.ttype is DML and token.value.upper() in {"SELECT", "INSERT", "UPDATE", "DELETE"}:
                    operation = token.value.upper()
                    continue

                if token.ttype is Whitespace:
                    continue

                if token.ttype is Punctuation:
                    fields.extend("all")
                    continue

                if isinstance(token, Identifier) and operation in {"INSERT", "UPDATE"} and table is None:
                    table = token.get_real_name()
                    continue

                if token.value.upper() == "SET" and table is not None:
                    fields.extend(parse.extract_set_clause(query))
                    continue

                if (isinstance(token, Function) or isinstance(token, Identifier) or isinstance(token, IdentifierList)) and len(fields)==0:
                    if operation in {"SELECT"}:
                        fields.extend(parse.extract_fields(token))
                        continue

                if token.value.upper() == "FROM":
                    table = parse.extract_tables_from_sql(query)
                    continue


            dependencies.append((operation, table, fields))

    return dependencies


def build_dependency_graph(sql_logs):
    """根据 SQL 执行日志构建依赖链路图"""
    G = nx.DiGraph()

    for log in sql_logs:
        sql_id = log['sql_id']
        dependencies = []
        extract_table_field_dependencies(log['sql'], dependencies)

        # 添加当前 SQL 节点
        G.add_node(sql_id, sql=log['sql'], tables=dependencies[0][1], fields=dependencies[0][2], duration=log['duration'])

        # 查找依赖的 SQL 语句（例如：子查询依赖）
        for dep_sql_id in log.get('dependencies', []):
            # 添加依赖关系
            G.add_edge(dep_sql_id, sql_id)

    return G

def draw_graph(G):
    """绘制依赖图"""
    pos = nx.spring_layout(G)  # 使用Spring布局绘制图形
    plt.figure(figsize=(12, 8))  # 设置图形大小
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=12, font_weight='bold')
    labels = nx.get_node_attributes(G, 'sql')  # 获取SQL查询作为标签
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("SQL Dependency Graph")
    plt.show()


