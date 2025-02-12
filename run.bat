
echo "Climate Change w/ COVID-19 Demo"
python -m cross_frame.frame --config configs/climate/covid-19/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/climate/covid-19/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/climate/covid-19/gemini-2.0-pro.yaml

echo "Feminism w/ COVID-19 Demo"
python -m cross_frame.frame --config configs/feminism/covid-19/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/feminism/covid-19/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/feminism/covid-19/gemini-2.0-pro.yaml

echo "Immigration w/ COVID-19 Demo"
python -m cross_frame.frame --config configs/immigration/covid-19/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/immigration/covid-19/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/immigration/covid-19/gemini-2.0-pro.yaml

