import os.path
import json

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

import re
import numpy as np
import glob
import pandas as pd

import torch
import numpy as np


def handleJavaCode(filename, code_range):
    """
    Extract his code and his comments for each node,
    1. filename is the original file;
    2. code-range is the range of this node
    """
    with open(filename, 'r') as f:
        file = f.read()
        file_list = file.replace("\t", " ").split("\n")
        range_file_list = []

        beginLine = code_range["beginLine"] - 2
        beginColumn = code_range["beginColumn"]
        endLine = code_range["endLine"] - 2
        endColumn = code_range["endColumn"]

        if beginLine < 0 or endLine < 0:
            return [], []
        if beginLine == endLine:
            for i in range(0, len(file_list)):
                if i == beginLine:
                    range_file_list.append(file_list[i][beginColumn - 1:endColumn])
        else:
            # print(len(file_list))
            for i in range(0, len(file_list)):
                if i == beginLine:
                    range_file_list.append(file_list[i][beginColumn - 1:])
                elif i == endLine:
                    range_file_list.append(file_list[i][0: endColumn])
                elif i > beginLine and i < endLine:
                    range_file_list.append(file_list[i])
            # print("kkk")

        nl_list = []
        code_list = []

        for str in range_file_list:
            if str.find("//") != -1:
                nl_list.append(str)
            elif str.find("*") != -1 and str.find("/*(MInterface)*/") == -1 and str.find("* 100 )") == -1 \
                    and str.find("t1.*, t2.*") == -1 and str.find("inner_query.*") == -1 and str.find(
                "SELECT *") == -1 and str.find("SELECT * ") == -1 \
                    and str.find("count(*)") == -1 and str.find("2.0 * ") == -1 and str.find(
                "bodyWeight * 2.0") == -1 and str.find("bodyWeight*-1)") == -1 \
                    and str.find(" - 1.0) * ") == -1 and str.find(")*(") == -1 and str.find(
                "/* Here we' go! */") == -1 and str.find(" * 12 )") == -1 \
                    and str.find("select *") == -1 and str.find("/*Notation.findNotation") == -1 and str.find(
                "*=") == -1 and str.find("* 100 )") == -1:
                nl_list.append(str)
            else:
                code_list.append(str)

        return nl_list, code_list


def codeEmbedding(nl_list, code_list, tokenizer, model):
    """
    CodeEmbedding the extracted data
    """
    # print("begin to embedding")

    code = ""
    nl = ""
    for str in code_list:
        code = code + str

    for str in nl_list:
        nl = nl + str

    code_tokens = tokenizer.tokenize(code)
    nl_tokens = tokenizer.tokenize(nl)
    token_list = []
    token_embeddings = []
    tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    token_list = cutToken(tokens, token_list)
    for token in token_list:
        token_id = tokenizer.convert_tokens_to_ids(token)
        context_embeddings = model(torch.tensor(token_id)[None, :])[0]
        token_embeddings.append(context_embeddings)

    torch_tensor = torch.cat(token_embeddings, dim=1)

    return torch_tensor.tolist()[0]


def cutToken(tokens, token_list):
    """
    Cut tokens which are too long
    """
    if len(tokens) > 500:
        token_list.append(tokens[0:500])
        tokens = tokens[500: len(tokens)]
        cutToken(tokens, token_list)
    else:
        token_list.append(tokens)
    return token_list


def one_hot_node_type(node_type):
    """
     Handle 68 kinds of nodes with One-Hot
    """
    node_type = node_type.replace("com.github.javaparser.ast.", "")

    hot_dict = {'ArrayCreationLevel': 0, 'CompilationUnit': 1, 'Modifier': 2, 'ClassOrInterfaceDeclaration': 3,
                'ConstructorDeclaration': 4, 'MethodDeclaration': 5, 'Parameter': 6,
                'VariableDeclarator': 7, 'BlockComment': 8, 'JavadocComment': 9,
                'LineComment': 10, 'ArrayAccessExpr': 11, 'ArrayCreationExpr': 12,
                'ArrayInitializerExpr': 13, 'AssignExpr': 14, 'BinaryExpr': 15,
                'BooleanLiteralExpr': 16, 'CastExpr': 17, 'CharLiteralExpr': 18, 'ClassExpr': 19,
                'ConditionalExpr': 20, 'DoubleLiteralExpr': 21, 'EnclosedExpr': 22,
                'FieldAccessExpr': 23, 'InstanceOfExpr': 24, 'IntegerLiteralExpr': 25,
                'LongLiteralExpr': 26, 'MarkerAnnotationExpr': 27, 'MemberValuePair': 28,
                'MethodCallExpr': 29, 'Name': 30, 'NameExpr': 31, 'NormalAnnotationExpr': 32,
                'NullLiteralExpr': 33, 'ObjectCreationExpr': 34, 'SimpleName': 35,
                'SingleMemberAnnotationExpr': 36, 'StringLiteralExpr': 37, 'SuperExpr': 38,
                'ThisExpr': 39, 'UnaryExpr': 40, 'VariableDeclarationExpr': 41,
                'AssertStmt': 42,
                'BlockStmt': 43, 'BreakStmt': 44, 'CatchClause': 45, 'ContinueStmt': 46,
                'DoStmt': 47, 'EmptyStmt': 48, 'ExplicitConstructorInvocationStmt': 49,
                'ExpressionStmt': 50, 'ForEachStmt': 51, 'ForStmt': 52, 'IfStmt': 53,
                'LabeledStmt': 54, 'LocalClassDeclarationStmt': 55, 'ReturnStmt': 56,
                'SwitchEntry': 57,
                'SwitchStmt': 58, 'ThrowStmt': 59, 'TryStmt': 60,
                'WhileStmt': 61,
                'ArrayType': 62,
                'ClassOrInterfaceType': 63, 'PrimitiveType': 64, 'TypeParameter': 65,
                'VoidType': 66,
                'WildcardType': 67, 'ElseStmt': 68, 'ElseIfStmt': 69, 'FinallyStmt': 70, 'CatchStmt': 71}

    index = hot_dict[node_type]
    all_zero = np.zeros(len(hot_dict.keys()), dtype=int)
    node_type_one_hot = all_zero.copy()
    node_type_one_hot[index] = 1
    # print(node_type_one_hot)
    return list(node_type_one_hot)


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.json")]


def write_pkl(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)


def empty_graph():
    return {"node_type": [], "node_range": [], "edge_list": [[], []], "edge_type": []}


def ConvertToGraph(json_content):
    STMT = ['AssertStmt', 'BreakStmt', 'ContinueStmt', 'DoStmt', 'ExplicitConstructorInvocationStmt',
            'ExpressionStmt', 'ForEachStmt', 'ForStmt', 'LabeledStmt', 'LocalClassDeclarationStmt', 'ReturnStmt',
            'SwitchEntry', 'SwitchStmt', 'ThrowStmt', 'WhileStmt', 'MethodDeclaration']
    EXPR = ['AssignExpr', 'MethodCallExpr', 'BinaryExpr', 'UnaryExpr', 'VariableDeclarationExpr', 'Parameter']
    COMMENT = ['JavadocComment', 'LineComment', 'BlockComment']
    TYPES = STMT + EXPR + COMMENT



    # 防御性检查JSON结构
    if not json_content.get("types") or \
       not isinstance(json_content["types"], list) or \
       len(json_content["types"]) == 0:
        return empty_graph()

    first_type = json_content["types"][0]
    members = first_type.get("members", [])
    if not isinstance(members, list) or len(members) == 0:
        return empty_graph()

    # 关键修改：遍历members找到第一个有效节点
    graph_raw = None
    for member in members:
        if isinstance(member, dict) and "!" in member and "range" in member:
            node_type = member["!"].split(".")[-1]
            if node_type in TYPES:
                graph_raw = member
                break
    if not graph_raw:
        return empty_graph()



    Vertice_type = []
    Vertice_info = []
    Edge_list = [[], []]
    Edge_type = []

    def loopKeys(graph, key, index):
        # 情况1： 节点是个list，这里就需要多重处理，因为相当于多个情况2
        if isinstance(graph[key], list):
            # graph[key]是个list，里面有好多个item
            for item in graph[key]:
                createGraph(item, key, index)

        # 情况2： 节点是个dict
        elif isinstance(graph[key], dict):
            createGraph(graph[key], key, index)

    def createGraph(graph, name, index):

        # 只有是节点！，才会递归
        if "!" in graph.keys():
            noneName = graph["!"].split(".")[-1]

            # 检查是否存在range字段
            if 'range' not in graph:
                # 记录缺失range的节点类型、父节点信息以及原始数据
                parent_info = f"Parent key: '{name}'" if name else "Root level"
                print(f"""
                [ERROR] Missing 'range' in node:
                - Type: {noneName}
                - Parent context: {parent_info}
                - Raw data: {json.dumps(graph, indent=2)}
                """)
                return  # 跳过该节点，避免崩溃


            if noneName in TYPES:
                Vertice_type.append(noneName)
                Vertice_info.append(graph["range"])
                node_id = len(Vertice_type) - 1
                if index != -1:
                    Edge_list[0].append(index)
                    Edge_list[1].append(node_id)
                    Edge_type.append("AST")

                for k in graph.keys():
                    loopKeys(graph, k, node_id)


            elif "TryStmt" in noneName:
                Vertice_type.append(noneName)
                try_range = graph["tryBlock"]["range"]
                try_range["beginColumn"] = try_range["beginColumn"] - 9
                Vertice_info.append(try_range)

                node_id = len(Vertice_type) - 1
                if index != -1:
                    Edge_list[0].append(index)
                    Edge_list[1].append(node_id)
                    Edge_type.append("AST")

                # 手动进入一层。。
                for key in graph.keys():
                    # 如果这个key是"catchClause"，然后又不为空
                    if "catchClause" in key:
                        for catch in graph["catchClauses"]:
                            subgraph = catch
                            Vertice_type.append("CatchStmt")
                            catch_range = subgraph["range"]
                            catch_range["beginColumn"] = catch_range["beginColumn"]
                            Vertice_info.append(catch_range)

                            sub_node_id = len(Vertice_type) - 1
                            if index != -1:
                                Edge_list[0].append(node_id)
                                Edge_list[1].append(sub_node_id)
                                Edge_type.append("AST")

                            for k in subgraph.keys():
                                loopKeys(subgraph, k, sub_node_id)

                    elif key == "finallyBlock":
                        subgraph = graph["finallyBlock"]
                        Vertice_type.append("FinallyStmt")

                        fin_range = subgraph["range"]
                        fin_range["beginColumn"] = fin_range["beginColumn"] - 25
                        Vertice_info.append(fin_range)

                        sub_node_id = len(Vertice_type) - 1
                        if index != -1:
                            Edge_list[0].append(node_id)
                            Edge_list[1].append(sub_node_id)
                            Edge_type.append("AST")

                        for k in subgraph.keys():
                            loopKeys(subgraph, k, sub_node_id)

                    elif key == "":
                        subgraph = graph[""]
                        Vertice_type.append("FinallyStmt")

                        fin_range = subgraph["range"]
                        fin_range["beginColumn"] = fin_range["beginColumn"] - 25
                        Vertice_info.append(fin_range)

                        sub_node_id = len(Vertice_type) - 1
                        if index != -1:
                            Edge_list[0].append(node_id)
                            Edge_list[1].append(sub_node_id)
                            Edge_type.append("AST")

                        for k in subgraph.keys():
                            loopKeys(subgraph, k, sub_node_id)

                    else:
                        loopKeys(graph, key, node_id)

            elif "IfStmt" in noneName:

                # 如果是if stmt，记住，这个最上面的if的范围是它的thenStmt

                if name == "elseStmt":
                    noneName = "ElseIfStmt"

                then_range = graph["thenStmt"]["range"]
                condition = graph["condition"]["range"]
                if_range = {"beginLine": condition["beginLine"], "beginColumn": condition["beginColumn"] - 8,
                            "endLine": then_range["endLine"], "endColumn": then_range["endColumn"]}
                Vertice_type.append(noneName)
                Vertice_info.append(if_range)

                node_id = len(Vertice_type) - 1
                if index != -1:
                    Edge_list[0].append(index)
                    Edge_list[1].append(node_id)
                    Edge_type.append("AST")

                for k in graph.keys():
                    loopKeys(graph, k, node_id)

            elif "BlockStmt" in noneName and name == "elseStmt":
                else_range = graph["range"]
                else_range["beginColumn"] = else_range["beginColumn"] - 9
                Vertice_info.append(else_range)
                Vertice_type.append("ElseStmt")

                node_id = len(Vertice_type) - 1
                if index != -1:
                    Edge_list[0].append(index)
                    Edge_list[1].append(node_id)
                    Edge_type.append("AST")

                for k in graph.keys():
                    loopKeys(graph, k, node_id)


            else:
                Vertice_type.append(noneName)
                Vertice_info.append(graph["range"])
                node_id = len(Vertice_type) - 1
                if index != -1:
                    Edge_list[0].append(index)
                    Edge_list[1].append(node_id)
                    Edge_type.append("AST")

                for k in graph.keys():
                    loopKeys(graph, k, node_id)

    createGraph(graph_raw, "graph", -1)

    return {"node_type": Vertice_type, "node_range": Vertice_info, "edge_list": Edge_list, "edge_type": Edge_type}


def json_parse_to_graph(R_PATHS_AST, U_PATHS_AST):
    """
       Convert json file to Preprocess Representation
    """
    # n_dataset_files = get_directory_files(N_PATHS_AST)
    r_dataset_files = get_directory_files(R_PATHS_AST)
    u_dataset_files = get_directory_files(U_PATHS_AST)

    graph_list = []
    target_list = []
    code_filename_list = []

    for json_file in r_dataset_files:
        with open(os.path.join(R_PATHS_AST, json_file)) as f:
            print(json_file)
            content = json.load(f)
            graph = ConvertToGraph(content)
            graph_list.append(graph)
            target_list.append(0)
            code_filename_list.append(os.path.join(R_PATHS_AST, json_file.replace(".json", ".java")))

    # for json_file in n_dataset_files:
    #     with open(os.path.join(N_PATHS_AST, json_file)) as f:
    #         print(json_file)
    #         content = json.load(f)
    #         graph = ConvertToGraph(content)
    #         graph_list.append(graph)
    #         target_list.append(1)
    #         code_filename_list.append(os.path.join(N_PATHS_AST, json_file.replace(".json", ".java")))

    for json_file in u_dataset_files:
        with open(os.path.join(U_PATHS_AST, json_file)) as f:
            print(json_file)
            content = json.load(f)
            graph = ConvertToGraph(content)
            graph_list.append(graph)
            target_list.append(1)
            code_filename_list.append(os.path.join(U_PATHS_AST, json_file.replace(".json", ".java")))

    return graph_list, target_list, code_filename_list


def graph_to_input(graph, fileName, target, tokenizer, model):
    """
       Convert Preprocess to Vector Data for train and test, adding extra Extra in this process
       这里的操作对象是一个图

    """
    node_type = graph["node_type"]  # node type
    node_range = graph["node_range"]  # node range

    edge_list = graph["edge_list"]
    edge_types = graph["edge_type"]

    # 保存着这个图的所有节点，上面是所有节点信息的embedding，下面是type的one hot
    node_embedding_list = []
    node_one_hot_list = []

    print("==================", fileName, "=============")

    # 在标数据流的时候有两种节点，一种是申明，一种是使用，申明都在前面
    node_declaration_list = []
    node_assign_list = []
    # 用来加控制流
    node_stmt_list = []
    # 包含comment所有
    all_node_list = []

    raw_code_list = []

    for i in range(len(node_range)):

        # 通过node range获得node在真实代码中是哪几行，并获得raw—code 以及 raw-code 通过bert后的结果
        code_range = node_range[i]
        nl_list, code_list = handleJavaCode(fileName, code_range)
        raw_code = nl_list + code_list
        raw_code_list.append(raw_code)  # 原代码
        node_embedding = codeEmbedding(nl_list, code_list, tokenizer, model)  # 通过bert后
        node_embedding_list.append(node_embedding)

        node_type_one_hot = one_hot_node_type(node_type[i])
        node_one_hot_list.append(node_type_one_hot)

        if 'AssignExpr' in node_type[i] \
                or 'MethodCallExpr' in node_type[i] \
                or 'BinaryExpr' in node_type[i] \
                or 'UnaryExpr' in node_type[i]:

            node_assign_list.append(
                [i, node_type[i], [code_range["beginLine"] - 1, code_range["endLine"] - 1],
                 re.split(' |\.|\)|\(|\[|\]|\=', "".join(code_list)), code_list])

            all_node_list.append(
                [i, node_type[i].split(".")[-1], [code_range["beginLine"] - 1, code_range["endLine"] - 1], code_list])

        elif 'VariableDeclarationExpr' in node_type[i] \
                or node_type[i] == 'Parameter':
            node_declaration_list.append(
                [i, node_type[i], [code_range["beginLine"] - 1, code_range["endLine"] - 1],
                 re.split(' |\.|\)|\(|\[|\]|\=', "".join(code_list)), code_list])

            all_node_list.append(
                [i, node_type[i], [code_range["beginLine"] - 1, code_range["endLine"] - 1], code_list])
        else:
            node_stmt_list.append(
                [i, node_type[i], [code_range["beginLine"] - 1, code_range["endLine"] - 1], code_list])
            all_node_list.append(
                [i, node_type[i], [code_range["beginLine"] - 1, code_range["endLine"] - 1], code_list])

    node_declaration_list.sort(key=lambda x: (x[2][0], x[2][1]))
    node_assign_list.sort(key=lambda x: (x[2][0], x[2][1]))
    node_stmt_list.sort(key=lambda x: (x[2][0], x[2][1]))
    all_node_list.sort(key=lambda x: (x[2][0], x[2][1]))

    # edge_types = []
    # edge_list = [[],[]]
    # # ADD LOGIC FLOWS
    # if len(all_node_list) > 1:
    #     for i in range(len(all_node_list) - 1):
    #         edge_list[0].append(all_node_list[i][0])
    #         edge_list[1].append(all_node_list[i + 1][0])
    #         edge_types.append("LOGIC")

    # data_edge_list = DataEdgeHandle(node_declaration_list, node_assign_list)
    # for data_edge in data_edge_list:
    #     edge_list[0].append(data_edge[0])
    #     edge_list[1].append(data_edge[1])
    #     edge_types.append("DATA")

    control_edge_list = [[], []]
    # # ADD CONTROL FLOWS
    # if len(node_stmt_list) > 1:
    #     for i in range(len(node_stmt_list) - 1):
    #         control_edge_list[0].append(node_stmt_list[i][0])
    #         control_edge_list[1].append(node_stmt_list[i + 1][0])
    #
    # remove_edge, add_edge = AddControlByHand(fileName, node_stmt_list)
    #
    # for edge in remove_edge:
    #     control_edge_list[0].remove(edge[0])
    #     control_edge_list[1].remove(edge[1])
    #
    # for edge in add_edge:
    #     control_edge_list[0].append(edge[0])
    #     control_edge_list[1].append(edge[1])
    #
    # for i in range(len(control_edge_list[0])):
    #     edge_list[0].append(control_edge_list[0][i])
    #     edge_list[1].append(control_edge_list[1][i])
    #     edge_types.append("CONTROL")

    # print(node_type)
    # print()
    # print(raw_code_list)
    # print()
    # print(edge_list)
    # print()
    # print(edge_types)

    return node_type, raw_code_list, node_embedding_list, edge_list, edge_types, target, node_one_hot_list


def DataEdgeHandle(declaration_list, assign_list):
    """
       Handle Extra Data Edge, three ways tested in Ablation Study
    """

    data_flow_edge_list = []

    # TYPE 1
    for decl in declaration_list:
        data_flow = []
        flag = False
        for assign in assign_list:
            if decl[3][1] in assign[3]:
                flag = True
                data_flow.append(assign[0])
        if flag:
            data_flow.insert(0, decl[0])
            for j in range(len(data_flow) - 1):
                data_flow_edge_list.append([data_flow[j], data_flow[j + 1]])

    # # TYPE 2
    # for decl in declaration_list:
    #     data_flow = []
    #     flag = False
    #     for assign in assign_list:
    #         if decl[2][1] in assign[2]:
    #             flag = True
    #             data_flow.append(assign[0])
    #     if flag:
    #         data_flow.insert(0, decl[0])
    #         for j in range(len(data_flow) - 1):
    #             data_flow_edge_list.append([data_flow[0], data_flow[j + 1]])
    #
    # # TYPE 3
    # for decl in declaration_list:
    #     data_flow = []
    #     flag = False
    #     for assign in assign_list:
    #         if decl[2][1] in assign[2]:
    #             flag = True
    #             data_flow.append(assign[0])
    #     if flag:
    #         data_flow.insert(0, decl[0])
    #         for j in range(len(data_flow) - 1):
    #             data_flow_edge_list.append([data_flow[j], data_flow[j + 1]])
    #             if [data_flow[0], data_flow[j + 1]] not in data_flow_edge_list:
    #                 data_flow_edge_list.append([data_flow[0], data_flow[j + 1]])

    # return data_flow_edge_list


def findIndex(stmt_node_list, line):
    for node in stmt_node_list:
        if node[2][0] == line:
            return node[0]



if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # N_PATHS_AST = "../Test/Neutral"
    R_PATHS_AST = "../Dataset/Readable"
    U_PATHS_AST = "../Dataset/Unreadable"
    model_path = os.path.join(os.path.dirname(__file__), 'codebert-base')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    graph_list, target_list, code_filename_list = json_parse_to_graph(R_PATHS_AST, U_PATHS_AST)

    graph_input = []
    file_input = []
    target_input = []
    graph_raw_code_nodes = []

    for i in range(len(graph_list)):

        # if "Scalabrino84.java" in code_filename_list[i]:
        node_type, raw_code_list, node_embedding_list, edge_list, edge_types, target, node_one_hot_list = graph_to_input(
            graph_list[i], code_filename_list[i], target_list[i], tokenizer, model)
        nodes_info = []

        # graph_raw_code_nodes.append({"graph_name": code_filename_list[i].split("/")[-1], "graph_nodes_codes": raw_code_list, "graph_nodes_type": node_type})
        graph_raw_code_nodes.append(code_filename_list[i].split("\\")[-1])

        
        

        for j in range(len(node_embedding_list)):
            node_embedding = np.array(node_embedding_list[j])
            node_embedding = np.mean(node_embedding_list[j], axis=0)
            node_info = np.concatenate((node_embedding.tolist(), node_one_hot_list[j]), axis=0)
            nodes_info.append(node_info)

        x = torch.tensor(nodes_info)
        print(x)
        x = x.to(torch.float32)
        # 动态调整填充矩阵的大小，确保至少能容纳当前图的节点数
        max_nodes = max(1000, x.size(0))  # 保留最小保障1000，但根据实际情况扩展
        x_zero = torch.zeros(max_nodes, 840).float()
        x_zero[:x.size(0), :] = x

        y = torch.tensor([target]).float()
        edge_index = torch.tensor(edge_list)
        graph_data = Data(x=x, edge_index=edge_index, y=target)
        target_input.append(target)
        # node_type #edge_type
        graph_input.append(graph_data)

        
    # print(graph_raw_code_nodes)    
    pkl_data = {"file": graph_raw_code_nodes, "input": graph_input, "target": target_input}
    cpg_dataset = pd.DataFrame(pkl_data)

    # please change the name ("input_XXXXXX.pkl") if necessary
    # the "matrix" is not necessary here, it's for future studying
    write_pkl(cpg_dataset[["input", "target"]], "../Dataset/Packaged Pkl/", f"input.pkl")
    print("Build pkl Successfully")
