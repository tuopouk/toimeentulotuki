# toimeentulotuki

Tämä sovellus hyödyntää Kelan julkaisemaa avointa dataa perustoimeentulotuen maksuista ja palautuksista sekä muodostaa niistä ennusteen.

Sovellus hyödyntää Ridge-regressiota ennusteen tekemiseen. Sovellus jakaa datan opetus, -validointi ja testidataan, minkä avulla optimoidaan regularisointiparametri lambda. Itse ennuste tehdään optimoidulla algoritmilla.

Sovellus löytyy täältä: https://toimeentulotuki.herokuapp.com/
Alkuperäinen datan lähde on täällä: https://www.avoindata.fi/data/fi/dataset/kelan-maksaman-perustoimeentulotuen-menot-ja-palautukset
