with open("test_draw.txt", "r") as f:
    content = f.read()
    # 去掉开头和结尾的中括号
    content = content.strip()[1:-1]
    # 按空格分割
    data = [float(x) for x in content.split()]
print(data)  # data 是一个 float 列表