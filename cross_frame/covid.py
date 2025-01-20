import ujson as json


problems = {
    "Confidence": "Trust in the security and effectiveness of vaccinations, the health authorities, and the health officials who recommend and develop vaccines.",
    "Complacency": "Complacency and laziness to get vaccinated due to low perceived risk of infections.",
    "Constraints": "Structural or psychological hurdles that make vaccination difficult or costly.",
    "Calculation": "Degree to which personal costs and benefits of vaccination are weighted.",
    "Collective Responsibility": "Willingness to protect others and to eliminate infectious diseases.",
    "Compliance": "Support for societal monitoring and sanctioning of people who are not vaccinated.",
    "Conspiracy": "Conspiracy thinking and belief in fake news related to vaccination.",
}


def load_covid(file_path: str):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data, problems
