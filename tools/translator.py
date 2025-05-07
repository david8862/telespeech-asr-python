#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" tool to translate text between different languages
Reference from:
https://www.cnblogs.com/geekbruce/articles/18361899
https://blog.csdn.net/weixin_41287260/article/details/122353715

install translate with following cmd:
pip install translate

install deep_translator with following cmd:
pip install deep_translator

install argostranslate with following cmd:
pip install argostranslate

NOTE: to have a better inference performance with large model (like 'large-v3-turbo'), you'd
      better to have a Nvidia GPU with >= 8GB Memory
"""
import os, sys, argparse
import glob
from tqdm import tqdm


def translate_translator(input_text_list, source_language, target_language, output_path):
    from translate import Translator
    translator = Translator(from_lang=source_language, to_lang=target_language)

    pbar = tqdm(total=len(input_text_list), desc='translate from %s to %s' % (source_language, target_language))
    for input_text_filename in input_text_list:
        text_file = open(input_text_filename, 'r', encoding='utf-8')
        text_lines = text_file.readlines()
        text_file.close()

        # open output file to write translated text
        text_file_basename = os.path.splitext(os.path.split(input_text_filename)[-1])[0]
        output_text_filename = os.path.join(output_path, text_file_basename + '.txt')
        output_text_file = open(output_text_filename, 'w', encoding='utf-8')

        for text_line in text_lines:
            # 执行翻译
            translated_text = translator.translate(text_line.strip())
            #print(translated_text)
            output_text_file.write(translated_text.lower())
            output_text_file.write('\n')
        output_text_file.close()
        pbar.update(1)
    pbar.close()
    return


def deeptrans_translator(input_text_list, source_language, target_language, output_path):
    from deep_translator import GoogleTranslator
    gt = GoogleTranslator()

    pbar = tqdm(total=len(input_text_list), desc='translate from %s to %s' % (source_language, target_language))
    for input_text_filename in input_text_list:
        text_file = open(input_text_filename, 'r', encoding='utf-8')
        text_lines = text_file.readlines()
        text_file.close()

        # open output file to write translated text
        text_file_basename = os.path.splitext(os.path.split(input_text_filename)[-1])[0]
        output_text_filename = os.path.join(output_path, text_file_basename + '.txt')
        output_text_file = open(output_text_filename, 'w', encoding='utf-8')

        for text_line in text_lines:
            # 执行翻译
            translated_text = gt.translate(text_line.strip(), target=target_language)
            #print(translated_text)
            output_text_file.write(translated_text.lower())
            output_text_file.write('\n')
        output_text_file.close()
        pbar.update(1)
    pbar.close()
    return


def argos_translator(input_text_list, source_language, target_language, output_path):
    import argostranslate.package
    import argostranslate.translate

    # 更新语言包索引
    argostranslate.package.update_package_index() # 注释掉这行代码，会加速变快。不然代码返回结果就较慢。
    # # 获取可用的语言包
    available_packages = argostranslate.package.get_available_packages()
    # 筛选出需要的语言包并安装
    package_to_install = next(
        filter(
            lambda x: x.from_code == source_language and x.to_code == target_language,
            available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    pbar = tqdm(total=len(input_text_list), desc='translate from %s to %s' % (source_language, target_language))
    for input_text_filename in input_text_list:
        text_file = open(input_text_filename, 'r', encoding='utf-8')
        text_lines = text_file.readlines()
        text_file.close()

        # open output file to write translated text
        text_file_basename = os.path.splitext(os.path.split(input_text_filename)[-1])[0]
        output_text_filename = os.path.join(output_path, text_file_basename + '.txt')
        output_text_file = open(output_text_filename, 'w', encoding='utf-8')

        for text_line in text_lines:
            # 执行翻译
            translated_text = argostranslate.translate.translate(text_line.strip(), source_language, target_language)
            #print(translated_text)
            output_text_file.write(translated_text.lower())
            output_text_file.write('\n')
        output_text_file.close()
        pbar.update(1)
    pbar.close()
    return



def main():
    parser = argparse.ArgumentParser(description='tool to translate text between different languages')
    parser.add_argument('--input_text_path', type=str, required=True,
                        help='file or directory for input .txt files')
    parser.add_argument('--source_language', type=str, required=False, default='en',
                        choices=[None, 'zh', 'en', 'fr', 'de', 'it', 'es', 'ja', 'ko', 'ru', 'tr', 'th'],
                        help = "Source language to translate. default=%(default)s")
    parser.add_argument('--target_language', type=str, required=False, default='zh',
                        choices=[None, 'zh', 'en', 'fr', 'de', 'it', 'es', 'ja', 'ko', 'ru', 'tr', 'th'],
                        help = "Target language to translate. default=%(default)s")
    parser.add_argument('--package_type', type=str, required=False, default='translate',
                        choices=['translate', 'deep_translator', 'argostranslate'],
                        help = "Python package to do the translation. default=%(default)s")
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save translated text files. default=%(default)s')
    args = parser.parse_args()

    # get .txt text file list or single .txt text file
    if os.path.isfile(args.input_text_path):
        input_text_list = [args.input_text_path]
    else:
        input_text_list = glob.glob(os.path.join(args.input_text_path, '*.txt'))

    os.makedirs(args.output_path, exist_ok=True)

    if args.package_type == 'translate':
        translate_translator(input_text_list, args.source_language, args.target_language, args.output_path)
    elif args.package_type == 'deep_translator':
        deeptrans_translator(input_text_list, args.source_language, args.target_language, args.output_path)
    elif args.package_type == 'argostranslate':
        argos_translator(input_text_list, args.source_language, args.target_language, args.output_path)
    else:
        print('Invalid package type:', args.package_type)
        return

    print('\nTranslate done. %s text files has been saved to %s' % (args.target_language, args.output_path))

if __name__ == "__main__":
    main()
