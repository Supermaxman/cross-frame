from datasets import load_dataset, concatenate_datasets


tweet_eval_problems = {
    "stance_abortion": {
        "Women's Rights": "Emphasizes abortion as a fundamental right and an aspect of gender equality, framing the decision as a personal one that the state should not interfere with.",
        "Fetal Rights": "Frames abortion as the taking of a human life, prioritizing the moral and legal rights of the fetus, often invoking the concept of personhood from conception.",
        "Sanctity of Life": "Portrays abortion as morally wrong under virtually all circumstances, often rooted in religious or ethical arguments about the sacredness of human life.",
        "Personal Morality": "Emphasizes that abortion is a complex moral decision that should be left to individual conscience rather than dictated by the state.",
        "Legality & Constitutional Rights": "Focuses on abortion as a constitutional right, particularly tied to privacy and liberty, and debates legal standards such as undue burden and personhood.",
        "Legislative & States' Rights": "Argues that abortion should be regulated by state legislatures rather than courts, often framed as restoring democratic decision-making on the issue.",
        "Health & Safety": "Positions abortion as a necessary medical procedure, critical for protecting women’s physical and mental health, and highlights the dangers of illegal or unsafe abortions.",
        "Abortion Harms Women": "Claims that abortion poses physical and psychological risks to women, sometimes invoking disputed medical evidence to support restrictions.",
        "Medical Necessity": "Debates whether abortion is ever truly necessary for maternal health, with pro-choice advocates emphasizing exceptions and pro-life advocates arguing modern medicine eliminates the need.",
        "Economic Consequences": "Frames abortion as economically essential for women’s financial stability and career prospects, particularly affecting low-income and marginalized groups.",
        "Abortion as Exploitation": "Argues that abortion disproportionately harms disadvantaged communities, sometimes framed as a form of racial or economic oppression.",
        "Gender Equality": "Asserts that access to abortion is necessary for women to have equal opportunities in education, employment, and public life.",
        "Motherhood & Traditional Roles": "Frames abortion as undermining the value of motherhood and suggests that true equality should involve supporting women as mothers rather than facilitating abortion.",
        "Reproductive Justice": "Expands the discussion beyond abortion rights to include racial, economic, and social factors, emphasizing the right to parent in safe and supportive conditions.",
        "Religious Freedom": "Argues that abortion laws should reflect religious beliefs about life, or conversely, that bans on abortion impose a religious viewpoint and violate church-state separation.",
        "Public Opinion & Political Polarization": "Examines how partisan divides and identity politics influence abortion attitudes, often reinforcing entrenched positions rather than changing minds.",
        "Language & Terminology": "Highlights how strategic word choices ('pro-choice' vs. 'pro-life', 'fetus' vs. 'baby') frame the debate and influence public perception.",
    },
    "stance_climate": {
        "Scientific Uncertainty & Climate Denial": "Framing climate science as uncertain or unsettled, casting doubt on the scientific consensus, or portraying climate change as a hoax. This reduces trust in experts and delays action.",
        "Ideological Polarization & Identity": "Presenting climate change as a partisan issue aligned with a specific political ideology, leading to social division and resistance based on group identity rather than scientific facts.",
        "Economic Trade-offs (Environment vs. Economy)": "Framing climate policies as harmful to economic growth, job security, and affordability, emphasizing short-term economic costs over long-term benefits or the costs of inaction.",
        "Elitist vs. Public Needs (Populist Framing)": "Portraying climate action as an agenda pushed by global elites, bureaucrats, and scientists, disconnected from the struggles of ordinary citizens. This frame suggests policies are unfair or imposed without grassroots support.",
        "Catastrophic 'Doomsday' Narratives & Fatalism": "Overemphasizing apocalyptic consequences of climate change without solutions, leading to public fear, helplessness, and disengagement due to perceived inevitability of disaster.",
        "Psychological Distance & Abstraction": "Framing climate change as a distant, abstract issue affecting future generations, remote places, or complex scientific processes, making it seem less relevant to people's immediate lives and concerns.",
    },
    "stance_feminist": {
        "Gender Equality vs. Misandry Myth": "The debate over whether feminism promotes gender equality for all or fosters hostility toward men. Supporters argue feminism seeks to end gender-based oppression, while critics claim it is anti-male and seeks female superiority.",
        "Victimhood vs. Empowerment": "The question of whether feminism empowers women by addressing systemic inequality or fosters a 'victim mentality.' Supporters argue that acknowledging discrimination is a step toward empowerment, while critics claim it discourages self-reliance and overemphasizes oppression.",
        "Traditional Gender Roles vs. Feminist Ideals": "The conflict between feminism and traditional gender roles. Feminists argue for freedom of choice in career and family life, while critics claim feminism devalues motherhood, homemaking, and femininity.",
        "Intersectionality vs. White Feminism": "The tension between feminism's goal of inclusivity across race, class, and sexuality versus critiques that mainstream feminism is dominated by white, Western, and middle-class perspectives, often ignoring marginalized groups.",
        "Economic Equality and the Pay Gap Debate": "The dispute over whether the gender wage gap is a result of discrimination or personal choices. Feminists argue that structural barriers cause persistent inequality, while critics claim the gap is exaggerated and primarily driven by women's career and lifestyle decisions.",
        "Policy Debates and Political Polarization": "The framing of feminist policies as either necessary for gender justice or as ideological overreach. Feminist goals such as reproductive rights and gender quotas are framed by supporters as essential for equality, while critics see them as attacks on traditional values or political extremism.",
        "Media Representation and Backlash": "The portrayal of feminism and feminists in the media, where supporters argue for accurate representation of feminist goals, while detractors frame feminists as extremists, humorless, or irrelevant. This problem includes media-driven backlash against feminist gains.",
    },
}


def load_tweet_eval(name: str):
    dataset = load_dataset("cardiffnlp/tweet_eval", name)
    test = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    # test = dataset["test"]
    test = test.filter(lambda x: x["label"] != 0)
    test = test.map(lambda x, idx: {"id": str(idx)}, with_indices=True)
    problems = tweet_eval_problems[name]
    test_examples = [
        {
            "id": x["id"],
            "text": x["text"],
        }
        for x in test
    ]
    return test_examples, problems


def load_abortion(file_path: str = None):
    return load_tweet_eval("stance_abortion")


def load_climate(file_path: str = None):
    return load_tweet_eval("stance_climate")


def load_feminist(file_path: str = None):
    return load_tweet_eval("stance_feminist")
