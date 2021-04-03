import glob, json, os, argparse
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
# from transformers import GPT2Tokenizer
from pathlib import Path



# check dataset without tokenization
FOR_REAL = True

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Params')
    # parser.add_argument('--segment_len', type=int, default=254,
    #                     help='the length of each example')
    # # we set this to be 254 instead of 256 because we want the input to be like: <control_code> input_ids <eos>
    # parser.add_argument('--stride', type=int, default=10,
    #                     help='stride to split training examples')
    # parser.add_argument('--dev_size', type=float, default=0.1,
    #                     help='split ratio of development set for each language')
    # args = parser.parse_args()

    # gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # gpt2_tok.to(device)
    # print ("\nEncoding using GPT2Tokenizer {}".format(gpt2_tok))
    # paths = ['Python', 'Java', 'Javascript']
    paths = ['Python']
    segments = {}


    def magic_path_filter(path):
        print ("We've received path {}".format(path))
        if not Path(path).exists():
            print (f"Sorry, we have no folder {Path(path)}")

        if path == "Python":
            # ret_str = glob.glob(f'{path}/**/*.py', recursive=True)
            ret_str = Path(path).rglob('*.py')
            return ret_str
        elif path == "Javascript":
            # ret_str = glob.glob(f'{path}/**/*.js', recursive=True)
            # ret_str = glob.glob(f'{path}/**/**/*.js', recursive=True)
            ret_str = Path(path).rglob('*.js')
            print ("Returning {}".format(ret_str))
            return ret_str
        else:
            # ret_str = glob.glob(f'{path}/**/*.java', recursive=True)
            ret_str = Path(path).rglob('*.java')
            return ret_str


    for path in paths:
        # source_files = glob.glob(magic_path_filter(path), recursive=True)
        source_files = magic_path_filter(path)
        # source_files = glob.glob(f'{path}/**/*.py' if path == "Python" else f'{path}/**/*.java', recursive=True)
        print ("Working with source_files {}".format(source_files))
        for each_src in tqdm(source_files):
            if os.path.isfile(each_src):
                
                # with open(each_src, "r", encoding="utf-8") as f:
                with open(each_src, "rb") as f:
                    print ("\nReading {}".format(each_src))
                    try:
                        code_content = f.read().decode('utf-8')
                        print (f"We have code {code_content} of len {len(code_content)}")
                    except UnicodeDecodeError as e:
                        print ("We have error {}. \nTrying to save with utf-8 encoding".format(e))
            
            if os.path.isdir(each_src):
                print ("Turns out {} is a directory. Processing...".format(each_src))
                subdir = Path(each_src).rglob('*.js')

                for each_sub_src in tqdm(subdir):
                    if os.path.isfile(each_sub_src):
                        # with open(each_sub_src, "r", encoding="utf-8") as f:
                        with open(each_sub_src, "rb") as f:
                            print ("\nReading {}".format(each_sub_src))
                            try:
                                code_content = f.read().decode('utf-8')
                            except UnicodeDecodeError as e:
                                print ("We have error {}. \nTrying to save with utf-8 encoding".format(e))
                            
                            
                    else:
                        print ("Still a directory. Go deeper?")                 

    