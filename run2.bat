
echo "COVID-19 w/ Immigration Demo"
python -m cross_frame.frame --config configs/covid-19/immigration/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/covid-19/immigration/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/covid-19/immigration/gemini-2.0-pro.yaml

echo "Abortion w/ Immigration Demo"
python -m cross_frame.frame --config configs/abortion/immigration/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/abortion/immigration/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/abortion/immigration/gemini-2.0-pro.yaml

echo "Climate Change w/ Immigration Demo"
python -m cross_frame.frame --config configs/climate/immigration/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/climate/immigration/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/climate/immigration/gemini-2.0-pro.yaml

echo "Feminism w/ Immigration Demo"
python -m cross_frame.frame --config configs/feminism/immigration/gemini-1.5-pro.yaml
python -m cross_frame.frame --config configs/feminism/immigration/gemini-2.0-flash.yaml
python -m cross_frame.frame --config configs/feminism/immigration/gemini-2.0-pro.yaml
