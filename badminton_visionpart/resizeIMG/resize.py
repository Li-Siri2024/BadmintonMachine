from PIL import Image

# 输入图片路径
input_path = "./image.jpg"   # 替换成你的图片文件名
output_path = "./resized_image.jpg"  # 可以覆盖原图，也可以另存为

# 目标尺寸：宽1340，高610
target_size = (610, 1340)

# 打开、resize、保存
with Image.open(input_path) as img:
    resized_img = img.resize(target_size, Image.ANTIALIAS)
    resized_img.save(output_path)
    print("图片已成功调整大小并保存为", output_path)

