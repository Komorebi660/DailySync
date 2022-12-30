import numpy as np
import csv
#import matplotlib.pyplot as plt

#100 country names, each with a unique tag(0-99)
country_list = ['Tuvalu', 'Paraguay', 'Egypt', 'Guyana', 'Dominica', 'Greenland', 'Lebanon', 'Georgia', 'Mozambique', 'Serbia', 'Croatia',
                'Bouvetoya', 'Uzbekistan', 'Myanmar', 'Eritrea', 'Bangladesh', 'Panama', 'Congo', 'Iran', 'Barbados', 'Iraq', 'Uganda',
                'Uruguay', 'Gambia', 'Niue', 'Tunisia', 'Bulgaria', 'Azerbaijan', 'Tajikistan', 'Benin', 'Canada', 'Burundi', 'Mauritius',
                'Spain', 'Cambodia', 'Djibouti', 'Hungary', 'Tanzania', 'Monaco', 'Singapore', 'Cameroon', 'India', 'Tonga', 'Ethiopia',
                'Madagascar', 'Morocco', 'Guatemala', 'Poland', 'Mayotte', 'Samoa', 'Cuba', 'Senegal', 'Jersey', 'Nauru', 'Australia',
                'Zimbabwe', 'Macao', 'Austria', 'Nicaragua', 'Suriname', 'Costa Rica', 'Ecuador', 'Haiti', 'Yemen', 'Mexico', 'Brazil',
                'Chad', 'Montenegro', 'Mongolia', 'Belize', 'Colombia', 'Timor-Leste', 'Germany', 'Malawi', 'Korea', 'Qatar', 'Bhutan',
                'Indonesia', 'Norway', 'Guam', 'Montserrat', 'Malta', 'Malvinas', 'Finland', 'Portugal', 'Maldives', 'Slovakia', 'Ireland',
                'Palau', 'Guinea', 'Somalia', 'Sweden', 'Iceland', 'Bahamas', 'Pakistan', 'Anguilla', 'Liechtenstein', 'American', 'Comoros',
                'Greece']

# zipf distribution
p = np.array([0.1928/r**1 for r in range(1, 100)]).astype(np.float64)
p = np.concatenate((p, [1.0-p.sum()]))


def gen_passage_filter():
    np.random.seed(100)
    tags = np.random.choice(np.arange(100), p=p, size=8841823)
    #plt.hist(tags, bins=100)
    #plt.show()
    #print(np.sum(tags==0), np.sum(tags==99))

    with open("passage_filter.tsv", "w", encoding="utf8") as f:
        for tag in tags:
            #tag, country
            f.write(f"{tag}\t{country_list[tag]}\n")


def gen_query_filter():
    np.random.seed(0)
    tags = np.random.choice(np.arange(100), p=p, size=6980)
    #print(np.sum(tags==0), np.sum(tags==99))

    with open("../data/queries.dev.small.tsv", "r", encoding="utf8") as f_query, \
            open("query_filter.tsv", "w", encoding="utf8") as f:
        tsvreader_query = csv.reader(f_query, delimiter="\t")
        idx = 0
        for [qid, _] in tsvreader_query:
            tag = tags[idx]
            #qid, tag, country
            f.write(f"{qid}\t{tag}\t{country_list[tag]}\n")
            idx += 1


def load_tsv(path):
    with open(path, "r", encoding="utf8") as f:
        tsvreader_query = csv.reader(f, delimiter="\t")
        idx = 0
        for [qid, tag, country] in tsvreader_query:
            print(qid, tag, country)
            idx += 1
            if idx == 10:
                break


if __name__ == "__main__":
    gen_passage_filter()
    gen_query_filter()
    # verify
    #load_tsv("query_filter.tsv")
