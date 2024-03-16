import pandas as pd
import torch

#if torch.backends.mps.is_available():
 #   print('not cpu')

# Load the dataset
data = pd.read_csv('implicit_hate_v1_stg0-2_posts.tsv', delimiter='\t')


en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

# Move model to GPU if available
#if torch.backends.mps.is_available():
 #   en2de.to(torch.device("mps"))
  #  de2en.to(torch.device("mps"))

quality_check = []
def balance_with_bt(df, num):
    i = 0
    while i < num:
        j = 0
        for post in df['post']:
            if i == num:
                break
            #translate
            translation = en2de.translate(post, temperature = 0.7)
            #translate back
            paraphrase = de2en.translate(translation, temperature = 0.7)
            # add paraphrase to new df row. copy other columns from post
      #      quality_check.append(('post: ', post, '\n', 't: ', translation, '\n', 'bt: ', paraphrase, '\n'))
            i += 1
            print(df['hate_or_not_hate'].iloc[j])
            new_row = {'post' : paraphrase, 
                    'hate_or_not_hate' : df['hate_or_not_hate'].iloc[j], 
                    'implicit_or_explicit' : df['implicit_or_explicit'].iloc[j], 
                    'implicit_class' : df['implicit_class'].iloc[j], 
                    'extra_implicit_class' : df['extra_implicit_class'].iloc[j]}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            j += 1
    return df

"""
hate 7349
not hate 13291

impl 6255
expl 1094

white gr 1507
incitement 1240
stereotypical 1105
inferiority 867
irony 795
threatening 666
other 79

1100 - 867 = 233
1100 - 795 = 305
1100 - 666 = 434
1100 - 79 = 1021

5000 - 1094 = 3906
"""

white_gr_df = data[data['implicit_class'] == 'white_grievance']
incit_df = data[data['implicit_class'] == 'incitement']
stereo_df = data[data['implicit_class'] == 'stereotypical']
inf_df = data[data['implicit_class'] == 'inferiority']
irony_df = data[data['implicit_class'] == 'irony']
threat_df = data[data['implicit_class'] == 'threatening']
other_ft = data[data['implicit_class'] == 'other']

expl_df = data[data['implicit_or_explicit'] == 'explicit_hate']

not_hate_df = data[data['hate_or_not_hate'] == 'not_hate']


#inf_df = balance_with_bt(inf_df, 233)
#irony_df = balance_with_bt(irony_df, 305)
#threat_df = balance_with_bt(threat_df, 434)
other_df = balance_with_bt(other_ft, 1021)


expl_df = balance_with_bt(expl_df, 3906)

missing_augmented_data = pd.concat([expl_df, other_df], ignore_index=True)

# Specify the file path for the TSV file
tsv_file_path = 'missing_augmented_data.tsv'

# Write the DataFrame to a TSV file
missing_augmented_data.to_csv(tsv_file_path, sep='\t', index=False)

# save quality check list to txt just because
# Specify the file path for the text file
#txt_file_path = 'quality_check.txt'

# Join the elements of the list with newline characters
#data = '\n'.join(quality_check)

# Write the joined string to the file
##with open(txt_file_path, 'w') as f:
   # f.write(data)
