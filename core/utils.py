"""
一些与算法无关、但被多个模块复用的工具函数。

当前文件主要提供：
- read_json: 读取 JSON 文件为 Python 对象
- write_json: 将 data(dict) 写入/追加到 JSON 文件

注意：
- write_json 的实现不是“先读出完整 JSON 再 merge 再写回”的常规做法，
  而是用文件指针在 JSON 末尾进行“增量追加”（通过手动处理逗号与大括号）。
  这种方式适用于仅追加顶层 key 的场景，但对并发写入、重复 key、以及更复杂 JSON 结构不友好。
"""

import json  # Python 标准库：JSON 序列化/反序列化
import os  # Python 标准库：文件/目录操作、路径与文件大小查询
import re  # Python 标准库：正则表达式（当前文件未使用，但可能为历史遗留）


def read_json(json_path):  # 读取指定路径的 JSON 文件并返回解析结果
    """
    读取 json_path 指向的 JSON 文件并返回解析结果。

    参数:
    - json_path: JSON 文件路径

    返回:
    - json_data: json.load() 的解析结果（通常为 dict 或 list）
    """
    with open(json_path, 'r') as f:  # 以只读模式打开 JSON 文件（默认文本模式）
        json_data = json.load(f)  # 从文件句柄读取并解析 JSON 为 Python 对象（dict/list 等）
    return json_data  # 返回解析后的 Python 对象，供调用方使用


def write_json(data, file_name, save_path="./", mode="r+"):  # 将 data 写入/追加到 save_path 下的 file_name.json
    """
    将 data(dict) 写入到 save_path/file_name.json。

    设计意图：
    - 如果文件不存在：创建空文件后，再以“追加顶层 key”的方式写入内容
    - 如果 mode == 'w'：直接覆盖写入整个 JSON（json.dump）
    - 其他情况：以“增量追加”的方式把 data 的每个 (key, value) 写入顶层

    参数:
    - data: 需要写入的 dict（顶层 key 将被写进 JSON 顶层）
    - file_name: 不含扩展名的文件名
    - save_path: 输出目录
    - mode:
      - 'w': 覆盖写入（会把 data 当作完整 JSON 写入）
      - 其他: 以追加方式逐条写入顶层 key
    """
    # 输出目录不存在就创建
    if not os.path.exists(save_path):  # 若输出目录不存在，则需要先创建
        os.makedirs(save_path)  # 递归创建目录（父目录不存在也会一并创建）
    # 拼出最终路径，并强制使用 .json 扩展名
    full_path = os.path.join(save_path, f"{file_name}.json")  # 生成最终输出文件路径（确保扩展名为 .json）

    if not os.path.exists(full_path):  # 若目标文件不存在，先创建一个空文件占位
        # 确保文件存在；先创建空文件，后续再用 r+ 写入
        with open(full_path, 'w') as file:  # 以写入模式创建文件（会覆盖同名文件，但这里文件不存在）
            pass  # 不写任何内容，使其成为空文件，后续再用 r+ 追加写入
    elif mode == "w":  # 覆盖写入模式：把 data 作为完整 JSON 写进去
        # 覆盖写入：直接序列化整个 data
        with open(full_path, 'w') as file:  # 以写入模式打开（会清空原内容）
            json.dump(data, file, indent=4)  # 将 data 序列化成 JSON 写入文件，并用缩进美化
        return  # 覆盖写入完成后直接返回，不再走“追加写入”逻辑

    # r+：可读可写，不会截断文件，适合做“尾部追加”
    with open(full_path, 'r+') as file:  # 以读写模式打开，不截断文件，便于在末尾手动追加
        # 逐条写入 data 的顶层项
        for key, value in data.items():  # 遍历 data 的每个顶层键值对，逐条写入 JSON 顶层
            if os.path.getsize(full_path) == 0:  # 若文件大小为 0，说明当前还是空文件
                # 文件为空时：先写入空对象 {}，再把指针移动到 '}' 前面插入第一条 key
                file.write('{}')  # 写入一个空 JSON 对象，后续再把第一条键值对插入到 '}' 之前
                file.seek(0, os.SEEK_END)  # 把指针移动到文件末尾，准备回退定位到 '}'
                file.seek(file.tell() - 1, os.SEEK_SET)  # 回退 1 个字符，让指针停在 '}' 之前
                file.write('\n')  # 写入换行，提升可读性
                file.write(json.dumps(key, indent=4))  # 写入 JSON 格式的 key（会自动加引号并转义）
                file.write(': ')  # 写入 key 与 value 之间的分隔符
                file.write(json.dumps(value, indent=4))  # 写入 JSON 格式的 value（可能是数字/列表/对象等）
                file.write('\n')  # 写入换行，保持结构清晰
                file.write('}')  # 补回 JSON 对象的结束大括号
                continue  # 第一条写入完成后进入下一条 key

            # 文件非空时：假设文件末尾是 "\n}" 或 "}"，先回退 2 个字符覆盖掉 '}\n' 的前一部分，
            # 插入逗号与新键值对后，再补回结束大括号。
            file.seek(0, os.SEEK_END)  # 将指针移动到文件末尾，准备在结尾追加新的键值对
            file.seek(file.tell() - 2, os.SEEK_SET)  # 回退 2 个字符，覆盖末尾的 "\n}" 或类似结尾
            file.write(',')  # 写入逗号，作为 JSON 顶层对象中新条目的分隔符
            file.write('\n')  # 写入换行，使追加内容换行展示
            file.write(json.dumps(key, indent=4))  # 写入新条目的 key
            file.write(': ')  # 写入 key/value 分隔符
            file.write(json.dumps(value, indent=4))  # 写入新条目的 value
            file.write('\n')  # 写入换行，随后补回结束大括号
            file.write('}')  # 写回 JSON 对象结束符，保持文件始终是合法 JSON
    return  # 函数结束（无返回值语义时返回 None），与上面覆盖写入路径一致
