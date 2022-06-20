import argparse


def generate_c_source(in_file_path, out_file_path, varname):
    with open(in_file_path, "rb") as infile, open(out_file_path, "a") as outfile:
        total = 0
        outfile.write("unsigned char __{}[] = {{\n".format(varname))
        while (content := infile.read(12)):
            total += len(content)
            outfile.write("\t{}".format(", ".join([format(int(byte), '#04x') for byte in content])))
            if infile.peek(1):
                outfile.write(",\n")
        outfile.write("\n}};\nunsigned int __{}_len = {};".format(varname, str(total)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='Path to the saved model or .h5 file.')
    parser.add_argument(
        '--output',
        type=str,
        help='Where to save the generated C source file.')
    parser.add_argument(
        "--varname",
        type=str,
        default="tflite",
        help="Name of the variable stored in the C file."
    )

    ARGS, unparsed = parser.parse_known_args()
    generate_c_source(ARGS.input, ARGS.output, ARGS.varname)