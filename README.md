# toimeentulotuki

Tämä sovellus hyödyntää Kelan julkaisemaa avointa dataa perustoimeentulotuen maksuista ja palautuksista sekä muodostaa niistä projektion. Ajatus on testata kuinka hyvin eri lineaarisen regression variantit soveltuvat toimeentulotuen kumulatiivisen kertymän ennustamiseen.

Sovellus hyödyntää käyttäjän mukaan joko Lasso-, Ridge-, tai Elastisen verkon regressiota projektion tekemiseen. Sovellus jakaa datan opetus, -validointi ja testidataan, minkä avulla optimoidaan regularisointiparametri lambda. Itse ennuste tehdään optimoidulla algoritmilla.

Käytetyt piirteet ovat edellisen päivän kumulatiivinen arvo sekä käänteinen etäisyys tuleviin maksupäiviin. Toimeentulotuen maksupäivät ovat Kelan mukaan kuukauden ensimmäinen pankkipäivä sekä 9., 16, ja 23. päivä tai näitä edeltävä pankkipäivä mikäli kyseinen päivä on pyhäpäivä. Käänteinen etäisyys on etäisyyden käänteisluku. Esim. jos maksupäivä on viiden päivän päästä, on käänteinen etäisyys 1/5. Tässä sovelluksessa erityistapaukset (etäisyys joko nolla tai yksi) on otettu huomioon siten, että etäisyyden ollessa nolla, käänteinen etäisyys määritellään ykkösenä. Muussa tapauksessa käänteinen etäisyys kerrotaan luvulla 0.75, jotta käänteinen etäisyys ei olisi sama etäisyyden ollessa yksi tai nolla.

Mukana on myös baseline-ratkaisuna tavallinen lineaarinen malli, jonka ainoa piirre on edellisen päivän kumulatiivinen arvo.

Elastista verkko on sovellettu siten, että Lasso/Ridge -splitti on 50-50.

Lineaariregressiot antavat melko hyvän ennusteen kumulatiiviselle datalle, mutta päiväkohtaiset ovat toistaiseksi heikkoja. Tätä saattanen parantaa toisella mallinnustavalla jatkossa.

Sovellus löytyy täältä: https://toimeentulotuki.herokuapp.com/
Alkuperäinen datan lähde on täällä: https://www.avoindata.fi/data/fi/dataset/kelan-maksaman-perustoimeentulotuen-menot-ja-palautukset
