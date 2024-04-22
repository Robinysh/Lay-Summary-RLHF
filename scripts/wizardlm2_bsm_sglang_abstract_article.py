import json

import jsonlines
import sglang as sgl
from tqdm import tqdm

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))


def write_lines_to_file(file_path, lines):
    """
    Write a list of lines to a file, with each element of the list as a separate line in the file.

    :param file_path: str, the path to the file to be written to
    :param lines: list of str, the lines of text to write to the file
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


@sgl.function
def multi_dimension_rephrase(s, metrics_prompts):
    """
    Use the SGLang framework to generate multiple lay summaries from a biomedical text, each targeting different readability, relevance, and factuality metrics specified in metrics_prompts and then merge them at the end.

    :param s: SGLang session object, maintains the state across multiple generations
    :param metrics_prompts: list of str, a list of prompts that guide the generation of summaries focusing on different aspects such as readability, relevance, and factuality.
    """
    s += sgl.system(
        "Rephrase the following text from the same biomedical paper to make it more accessible and understandable to non-expert audiences, commonly referred to as 'lay summaries'."
    )
    forks = s.fork(len(metrics_prompts))
    for i in range(len(metrics_prompts)):
        forks[i] += sgl.user(metrics_prompts[i] + "\n Answer:")
        forks[i] += sgl.assistant(
            sgl.gen("generate", max_tokens=2048, temperature=0.8, top_p=0.95)
        )
    forks.join()
    s += sgl.user(
        "USER: You must merge the three summaries into a single summary such that the final lay summary of the paper is readable, relevant, and factual. Make sure that the merged summary is coherent and contains all important concepts from each summary that such that it is easily understandable to a 10th grade student. Please keep the merged summary to one long paragraph and make sure that it is factual, readable, relevant, and coherent. Summaries:"
    )
    for i in range(len(metrics_prompts)):
        s += f"Summary {i + 1}:\n" + forks[i]["generate"] + "\n"
    s += "In summary: ASSISTANT </s>"
    s += sgl.assistant(sgl.gen("merge", max_tokens=2048, temperature=0.8, top_p=0.95))


def rephrase_texts(abstract, article):
    metrics_prompts = [
        f"USER: Given the abstract of a biomedical paper, generate a summary paragraph that is easy to read and understand. Use simple language and short sentences. The summary should be suitable for a broad audience, ensuring that it is accessible to 12th grade readers. Focus on clarity and conciseness. \nAbstract: {abstract} \n ASSISTANT: below is a readable lay summary of this abstract: </s>",
        f"""
        USER: Given the abstract of a biomedical paper, generate a summary paragraph that closely aligns with the key themes and facts of the original text. Concentrate on capturing the most critical content to ensure high overlap of unigrams, overlap of bigrams, and longest common subsequences. Utilize semantic richness to ensure the summary maintains the meaning and context of the source material.
        Here are some examples of abstract and lay summary pairs:

        USER: Abstract: In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions . Death rates in men and women ≥ 45 years peaked in December to February and were lowest in June to August , driven by cardiorespiratory diseases and injuries . In these ages , percent difference in death rates between peak and minimum months did not vary across climate regions , nor changed from 1980 to 2016 . Under five years , seasonality of all-cause mortality largely disappeared after the 1990s . In adolescents and young adults , especially in males , death rates peaked in June/July and were lowest in December/January , driven by injury deaths .

        Lay summary of abstract: ASSISTANT: In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective . Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death . Now , Parks et al . show that there are age and sex differences in which times of year most deaths occur . Parks et al . analyzed data on US deaths between 1980 and 2016 . While overall deaths in a year were highest in winter and lowest in summer , a greater number of young men died during summer – mainly due to injuries – than during winter . Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller . Deaths among women and men aged 45 or older peaked between December and February – largely caused by respiratory and heart diseases , or injuries . Deaths in this older age group were lowest during the summer months . Death patterns in older people changed little over time . No regional differences were found in seasonal death patterns , despite large climate variation across the USA . The analysis by Parks et al . suggests public health and medical interventions have been successful in reducing seasonal deaths among many groups . But more needs to be done to address seasonal differences in deaths among older adults . For example , by boosting flu vaccination rates , providing warnings about severe weather and better insulation for homes . Using technology like hands-free communication devices or home visits to help keep vulnerable elderly people connected during the winter months may also help . </s>

        USER: Abstract: Kidney function depends on the nephron , which comprises a blood filter , a tubule that is subdivided into functionally distinct segments , and a collecting duct . How these regions arise during development is poorly understood . The zebrafish pronephros consists of two linear nephrons that develop from the intermediate mesoderm along the length of the trunk . Here we show that , contrary to current dogma , these nephrons possess multiple proximal and distal tubule domains that resemble the organization of the mammalian nephron . We examined whether pronephric segmentation is mediated by retinoic acid ( RA ) and the caudal ( cdx ) transcription factors , which are known regulators of segmental identity during development . Inhibition of RA signaling resulted in a loss of the proximal segments and an expansion of the distal segments , while exogenous RA treatment induced proximal segment fates at the expense of distal fates . Loss of cdx function caused abrogation of distal segments , a posterior shift in the position of the pronephros , and alterations in the expression boundaries of raldh2 and cyp26a1 , which encode enzymes that synthesize and degrade RA , respectively . These results suggest that the cdx genes act to localize the activity of RA along the axis , thereby determining where the pronephros forms . Consistent with this , the pronephric-positioning defect and the loss of distal tubule fate were rescued in embryos doubly-deficient for cdx and RA . These findings reveal a novel link between the RA and cdx pathways and provide a model for how pronephric nephrons are segmented and positioned along the embryonic axis .

        Lay summary of abstract: ASSISTANT: In the kidney , structures known as nephrons are responsible for collecting metabolic waste . Nephrons are composed of a blood filter ( glomerulus ) followed by a series of specialized tubule regions , or segments , which recover solutes such as salts , and finally terminate with a collecting duct . The genetic mechanisms that establish nephron segmentation in mammals have been a challenge to study because of the kidney's complex organogenesis . The zebrafish embryonic kidney ( pronephros ) contains two nephrons , previously thought to consist of a glomerulus , short tubule , and long stretch of duct . In this study , we have redefined the anatomy of the zebrafish pronephros and shown that the duct is actually subdivided into distinct tubule segments that are analogous to the proximal and distal segments found in mammalian nephrons . Next , we used the zebrafish pronephros to investigate how nephron segmentation occurs . We found that retinoic acid ( RA ) induces proximal pronephros segments and represses distal segment fates . Further , we found that the caudal ( cdx ) transcription factors direct the anteroposterior location of pronephric progenitors by regulating the site of RA production . Taken together , these results reveal that a cdx-RA pathway plays a key role in both establishing where the pronephros forms along the embryonic axis as well as its segmentation pattern . </s>

        USER: Abstract: Chemotherapeutic resistance in triple-negative breast cancer ( TNBC ) has brought great challenges to the improvement of patient survival . The mechanisms of taxane chemoresistance in TNBC have not been well investigated . Our results illustrated C-C motif chemokine ligand 20 ( CCL20 ) was significantly elevated during taxane-containing chemotherapy in breast cancer patients with nonpathologic complete response . Furthermore , CCL20 promoted the self-renewal and maintenance of breast cancer stem cells ( BCSCs ) or breast cancer stem-like cells through protein kinase Cζ ( PKCζ ) or p38 mitogen-activated protein kinase ( MAPK ) -mediated activation of p65 nuclear factor kappa B ( NF-κB ) pathway , significantly increasing the frequency and taxane resistance of BCSCs . Moreover , CCL20-promoted NF-κB activation increased ATP-binding cassette subfamily B member 1 ( ABCB1 ) /multidrug resistance 1 ( MDR1 ) expression , leading to the extracellular efflux of taxane . These results suggested that chemotherapy-induced CCL20 mediated chemoresistance via up-regulating ABCB1 . In addition , NF-κB activation increased CCL20 expression , forming a positive feedback loop between NF-κB and CCL20 pathways , which provides sustained impetus for chemoresistance in breast cancer cells . Our results suggest that CCL20 can be a novel predictive marker for taxane response , and the blockade of CCL20 or its downstream pathway might reverse the taxane resistance in breast cancer patients .

        Lay summary of abstract: ASSISTANT: Breast cancers are divided into several different subtypes according to the expression of three receptors: estrogen receptor ( ER ) , progesterone receptor ( PR ) , and human epidermal receptor 2 ( HER2 ) . One subtype of breast cancer is known as triple-negative breast cancer ( TNBC ) because of lack of expression of all three receptors . TNBC is highly malignant and susceptible to developing drug resistance during chemotherapy . However , the reasons for resistance have not been well explained . We found that a secretory protein called C-C motif chemokine ligand 20 ( CCL20 ) responds to taxane chemotherapy in TNBC and promotes the expression of a drug efflux pump through a specific signaling pathway ( CCL20-protein kinase Cζ [PKCζ]/p38 mitogen-activated protein kinase [MAPK]-nuclear factor kappa B [NF-κB]-ATP-binding cassette subfamily B member 1 [ABCB1] ) . Increasing levels of CCL20 result in expanding breast cancer stem-like cell population and chemotherapeutic resistance in breast cancer cells . Breast cancer stem-like cells are a special population of cells with self-renewal and differentiation potential within the tumor , which are often the culprit of tumor reoccurrence , drug resistance , and metastasis . Our results suggest that blocking CCL20 or its downstream signal can significantly improve the therapeutic efficacy of TNBC . </s>

        USER: Abstract: {abstract}
        Lay summary of abstract: ASSISTANT: </s> """,
        f"""
        USER: Carefully summarize this biomedical article, ensuring the summary is factually accurate and closely adheres to the presented data and conclusions. Focus on summarizing key findings, methodologies, and conclusions. Verify that each statement in your summary can be directly traced back to specific sections or data in the article. Please avoid adding interpretations or information not explicitly mentioned in the text.

        USER: Article: {article}
        Lay summary of article: ASSISTANT: </s> """,
    ]
    state = multi_dimension_rephrase.run(metrics_prompts)
    return state.messages()[-1]["content"]


if __name__ == "__main__":
    DATA_FOLDER = "/data/colx531/biolaysumm2024_data/"
    file_names = ["PLOS_test.jsonl", "eLife_test.jsonl"]

    for file_name in file_names:
        with open(DATA_FOLDER + file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item["article"].split("\n")
            item["sections"] = dict(zip(item["headings"], sections))

        predictions = []
        with jsonlines.open(
            f'prediction_sglang_bsm_abstract_article_WizardLM2_7B_{file_name.split(".", maxsplit=1)[0]}.jsonl',
            mode="w",
        ) as writer:
            for item in tqdm(data, leave=True):
                abstract = item["sections"]["Abstract"]
                article = item["article"]
                rephrased_abstract = rephrase_texts(abstract, article).strip()
                writer.write({"id": item["id"], "prediction": rephrased_abstract})
