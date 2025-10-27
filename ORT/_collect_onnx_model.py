import os
import onnx
import shutil


if __name__ == '__main__':
    this_dir = "../nnsmith/onnx_data/onnx_nnsmith_2w"
    for cnt, sub_dir_name in enumerate(os.listdir(this_dir)):
        # print(sub_dir)
        sub_dir = os.path.join(this_dir, sub_dir_name)
        # print(sub_dir)
        onnx_file_path = os.path.join(sub_dir, "model.onnx")
        new_path = f"../nnsmith/onnx_data/onnx_nnsmith/{cnt}.onnx"
        print(new_path)
        try:
            mm = onnx.load(onnx_file_path)
            onnx.checker.check_model(mm)
            shutil.copyfile(onnx_file_path, new_path)
        except Exception as e:
            print(e)
