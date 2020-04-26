import os, requests, gzip, shutil, numpy as np
from tqdm import tqdm

def get_number(byte):
    return int.from_bytes(byte, "big")

def parse_images(file_path):
    with open(file_path, "rb") as f:
        magic_num = get_number(f.read(4))
        image_count = get_number(f.read(4))
        rows = get_number(f.read(4))
        cols = get_number(f.read(4))

        images = np.zeros((image_count, rows, cols))

        current_img = 0
        current_row = 0
        current_col = 0

        filename = file_path.split("/")[-1]
        pbar = tqdm(total=image_count, desc= f"Parsing file {filename}")
        while current_img < image_count:
            num = get_number(f.read(1))
            images[current_img, current_row, current_col] = num

            if current_row == rows - 1 and current_col == cols - 1:
                current_img += 1
                current_row = 0
                current_col = 0
                pbar.update()

            elif current_col == cols - 1:
                current_row += 1
                current_col = 0
            else:
                current_col += 1
    return images

def parse_labels(file_path):
    with open(file_path, "rb") as f:
        magic_num = get_number(f.read(4))
        label_count = get_number(f.read(4))

        labels = np.zeros((label_count))
        current_label = 0

        filename = file_path.split("/")[-1]
        pbar = tqdm(total=label_count, desc= f"Parsing file {filename}")

        while current_label < label_count:
            num = get_number(f.read(1))
            labels[current_label] = num
            current_label += 1
            pbar.update()

        return labels

def unpack_and_replace(data_path):
    for file_name in urls.keys():
        file_path = os.path.join(data_path, file_name) + ".gz"
        with gzip.open(file_path, "rb") as f_in:
            with open(file_path[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)

def convert_to_ndarray_files(data_path):
    for file_name in urls.keys():
        file_path = os.path.join(data_path, file_name)
        if file_name[0] == "x":
            npy = parse_images(file_path)
        else:
            npy = parse_labels(file_path)
        np.save(file_path + ".npy", npy)
        os.remove(file_path)

def parse_from_unpacked_files(data_path):
    x_train = parse_images(os.path.join(data_path, "x_train"))
    y_train = parse_labels(os.path.join(data_path, "y_train"))

    x_test = parse_images(os.path.join(data_path, "x_test"))
    y_test = parse_labels(os.path.join(data_path, "y_test"))

    return (x_train, y_train), (x_test, y_test)

def parse_from_ndarray_files(data_path):
    x_train = np.load(os.path.join(data_path, "x_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))

    x_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    return (x_train, y_train), (x_test, y_test)

urls = {
    "x_train" : "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "y_train" : "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "x_test" : "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "y_test" : "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_file(url, file_path):
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        filename = url.split("/")[-1]
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length']), desc= f"Downloading file {filename}")
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return file_path

def download_data(data_path):
        for file_name, url in urls.items():
            file_path = os.path.join(data_path, file_name) + ".gz"
            download_file(url, file_path)

def load_data(data_path):
    try:
        os.makedirs(data_path)
        download_data(data_path)
        unpack_and_replace(data_path)
        convert_to_ndarray_files(data_path)
        return parse_from_ndarray_files(data_path)
    except FileExistsError:
        return parse_from_ndarray_files(data_path)  