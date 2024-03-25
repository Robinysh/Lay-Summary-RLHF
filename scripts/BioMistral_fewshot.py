import jsonlines
import json
from openai import OpenAI
from icecream import ic
from tqdm import tqdm
from huggingface_hub import InferenceClient

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



def write_lines_to_file(file_path, lines):
    """
    Write a list of lines to a file, with each element of the list as a separate line in the file.

    :param file_path: str, the path to the file to be written to
    :param lines: list of str, the lines of text to write to the file
    """
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def rephrase_texts(text):
    prompt = f'''
    <s>[INST]
    Rephrase the following abstract from a medical paper to make it more accessible and understandable to non-expert audiences, commonly referred to as "lay summaries".
    These lay summaries aim to present the key information from the original articles in a way that is less technical and contains more background information,
    making it easier for a broader range of readers, including researchers, medical professionals, journalists, and the general public, to comprehend the content.

    You should keep the words simple and the sentences short.
    You should also avoid using jargon and technical terms.
    The summary shold be easily understood by a 12th grade student.
    You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones.

    This is very important to my career. You'd better be sure.

    Here are some examples of abstract and lay summary pairs:

    Abstract: In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions . Death rates in men and women ≥ 45 years peaked in December to February and were lowest in June to August , driven by cardiorespiratory diseases and injuries . In these ages , percent difference in death rates between peak and minimum months did not vary across climate regions , nor changed from 1980 to 2016 . Under five years , seasonality of all-cause mortality largely disappeared after the 1990s . In adolescents and young adults , especially in males , death rates peaked in June/July and were lowest in December/January , driven by injury deaths .

    Lay summary of abstract: [/INST]In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective . Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death . Now , Parks et al . show that there are age and sex differences in which times of year most deaths occur . Parks et al . analyzed data on US deaths between 1980 and 2016 . While overall deaths in a year were highest in winter and lowest in summer , a greater number of young men died during summer – mainly due to injuries – than during winter . Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller . Deaths among women and men aged 45 or older peaked between December and February – largely caused by respiratory and heart diseases , or injuries . Deaths in this older age group were lowest during the summer months . Death patterns in older people changed little over time . No regional differences were found in seasonal death patterns , despite large climate variation across the USA . The analysis by Parks et al . suggests public health and medical interventions have been successful in reducing seasonal deaths among many groups . But more needs to be done to address seasonal differences in deaths among older adults . For example , by boosting flu vaccination rates , providing warnings about severe weather and better insulation for homes . Using technology like hands-free communication devices or home visits to help keep vulnerable elderly people connected during the winter months may also help .

    [INST]Abstract: Kidney function depends on the nephron , which comprises a blood filter , a tubule that is subdivided into functionally distinct segments , and a collecting duct . How these regions arise during development is poorly understood . The zebrafish pronephros consists of two linear nephrons that develop from the intermediate mesoderm along the length of the trunk . Here we show that , contrary to current dogma , these nephrons possess multiple proximal and distal tubule domains that resemble the organization of the mammalian nephron . We examined whether pronephric segmentation is mediated by retinoic acid ( RA ) and the caudal ( cdx ) transcription factors , which are known regulators of segmental identity during development . Inhibition of RA signaling resulted in a loss of the proximal segments and an expansion of the distal segments , while exogenous RA treatment induced proximal segment fates at the expense of distal fates . Loss of cdx function caused abrogation of distal segments , a posterior shift in the position of the pronephros , and alterations in the expression boundaries of raldh2 and cyp26a1 , which encode enzymes that synthesize and degrade RA , respectively . These results suggest that the cdx genes act to localize the activity of RA along the axis , thereby determining where the pronephros forms . Consistent with this , the pronephric-positioning defect and the loss of distal tubule fate were rescued in embryos doubly-deficient for cdx and RA . These findings reveal a novel link between the RA and cdx pathways and provide a model for how pronephric nephrons are segmented and positioned along the embryonic axis .

    Lay summary of abstract: [/INST]In the kidney , structures known as nephrons are responsible for collecting metabolic waste . Nephrons are composed of a blood filter ( glomerulus ) followed by a series of specialized tubule regions , or segments , which recover solutes such as salts , and finally terminate with a collecting duct . The genetic mechanisms that establish nephron segmentation in mammals have been a challenge to study because of the kidney's complex organogenesis . The zebrafish embryonic kidney ( pronephros ) contains two nephrons , previously thought to consist of a glomerulus , short tubule , and long stretch of duct . In this study , we have redefined the anatomy of the zebrafish pronephros and shown that the duct is actually subdivided into distinct tubule segments that are analogous to the proximal and distal segments found in mammalian nephrons . Next , we used the zebrafish pronephros to investigate how nephron segmentation occurs . We found that retinoic acid ( RA ) induces proximal pronephros segments and represses distal segment fates . Further , we found that the caudal ( cdx ) transcription factors direct the anteroposterior location of pronephric progenitors by regulating the site of RA production . Taken together , these results reveal that a cdx-RA pathway plays a key role in both establishing where the pronephros forms along the embryonic axis as well as its segmentation pattern .

    [INST]Abstract: Chemotherapeutic resistance in triple-negative breast cancer ( TNBC ) has brought great challenges to the improvement of patient survival . The mechanisms of taxane chemoresistance in TNBC have not been well investigated . Our results illustrated C-C motif chemokine ligand 20 ( CCL20 ) was significantly elevated during taxane-containing chemotherapy in breast cancer patients with nonpathologic complete response . Furthermore , CCL20 promoted the self-renewal and maintenance of breast cancer stem cells ( BCSCs ) or breast cancer stem-like cells through protein kinase Cζ ( PKCζ ) or p38 mitogen-activated protein kinase ( MAPK ) -mediated activation of p65 nuclear factor kappa B ( NF-κB ) pathway , significantly increasing the frequency and taxane resistance of BCSCs . Moreover , CCL20-promoted NF-κB activation increased ATP-binding cassette subfamily B member 1 ( ABCB1 ) /multidrug resistance 1 ( MDR1 ) expression , leading to the extracellular efflux of taxane . These results suggested that chemotherapy-induced CCL20 mediated chemoresistance via up-regulating ABCB1 . In addition , NF-κB activation increased CCL20 expression , forming a positive feedback loop between NF-κB and CCL20 pathways , which provides sustained impetus for chemoresistance in breast cancer cells . Our results suggest that CCL20 can be a novel predictive marker for taxane response , and the blockade of CCL20 or its downstream pathway might reverse the taxane resistance in breast cancer patients .

    Lay summary of abstract: [/INST]Breast cancers are divided into several different subtypes according to the expression of three receptors: estrogen receptor ( ER ) , progesterone receptor ( PR ) , and human epidermal receptor 2 ( HER2 ) . One subtype of breast cancer is known as triple-negative breast cancer ( TNBC ) because of lack of expression of all three receptors . TNBC is highly malignant and susceptible to developing drug resistance during chemotherapy . However , the reasons for resistance have not been well explained . We found that a secretory protein called C-C motif chemokine ligand 20 ( CCL20 ) responds to taxane chemotherapy in TNBC and promotes the expression of a drug efflux pump through a specific signaling pathway ( CCL20-protein kinase Cζ [PKCζ]/p38 mitogen-activated protein kinase [MAPK]-nuclear factor kappa B [NF-κB]-ATP-binding cassette subfamily B member 1 [ABCB1] ) . Increasing levels of CCL20 result in expanding breast cancer stem-like cell population and chemotherapeutic resistance in breast cancer cells . Breast cancer stem-like cells are a special population of cells with self-renewal and differentiation potential within the tumor , which are often the culprit of tumor reoccurrence , drug resistance , and metastasis . Our results suggest that blocking CCL20 or its downstream signal can significantly improve the therapeutic efficacy of TNBC .

    [INST]Abstract: {text}

    Lay summary of abstract: [/INST]'''

    completion = openai_client.completions.create(
        model="BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM",
        prompt=prompt,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95
    )
    ## Assuming the response contains a field 'choices' with the rephrased text
    rephrased_text = completion.choices[0].text.strip()
    return rephrased_text


if __name__ == '__main__':

    data_folder = '/data/colx531/biolaysumm2024_data/'
    file_names = ['PLOS_test.jsonl', 'eLife_test.jsonl']

    for file_name in file_names:
        with open(data_folder+file_name, 'r') as f:
            data = [json.loads(line) for line in f]

        for item in data:
            sections = item['article'].split('\n')
            item['sections'] = {k: v for k, v in zip(item['headings'], sections)}
        
        # Modify OpenAI's API key and API base to use vLLM's API server.
        predictions = []
        with jsonlines.open(f'prediction_fewshot_BioMistral7B_{file_name.split(".")[0]}.jsonl', mode='w') as writer:
            for item in tqdm(data, leave=True):
                # Extract abstracts
                abstract = item['sections']['Abstract']
                # Rephrase abstracts
                rephrased_abstract = rephrase_texts(abstract)
                writer.write({'id': item['id'], 'prediction': rephrased_abstract})
