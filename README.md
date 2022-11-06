## VKMKE

Riznica za predmet višeg kursa konačnih elemenata na master studijama Građevinskog fakulteta u Beogradu.

### Sadržaj:
1. [Instalacija Python3 i potrebnih paketa](#instalacija-python3-i-potrebnih-paketa)
2. [Instalacija Git-a](#instalacija-gita)
3. [Kloniranje riznice](#kloniranje-riznice)
4. [Pokretanje skripte](#pokretanje-skripte)
5.  [Wiki](https://codeberg.org/nikolal/VKMKE/wiki)
6. [Kontakt](#kontakt)

### Instalacija Python3 i potrebnih paketa

#### Windows:

Skini Python sa ovog [linka](https://www.python.org/ftp/python/3.9.4/python-3.9.4-amd64.exe) prilikom instaliranja čekiraj "*Add Python3.9 to PATH*" i nastavi sa instalacijom. Ako na kraju instalacije pita za *path lenght* klikni na plavi tekst pa potom na cancel.
Nakon uspešne instalacije potrebno je dodati pakete pod nazivom [sympy](https://www.sympy.org/en/index.html), [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/) i [matplotlib](https://matplotlib.org/). To radiš na sledeći nacin:

1. Otvori *Powershell* desnim klikom na start pa klikni na *Windows Powershell*
2. Verovatno ćes morati da apdejtuješ *pip*, to radiš tako što nalepiš sledeći tekst u Powershell i stisneš enter: <br> 
`python -m pip install --upgrade pip`
3. Nakon apdejtovanog *pip-a* nalepi sledeći tekst da bi instalirao *sympy pandas*: <br>
`pip install numpy sympy pandas matplotlib`

#### Linux:
Sigurno imaš već Python, nije potrebno da bude najnovija 3.9x verzija pa su ti potrebni samo *sympy* i *pandas* koje instaliraš sa sledećom komandom u terminalu: <br>
`pip3 install numpy sympy pandas matplotlib`

### Instalacija Gita

[Git](https://en.wikipedia.org/wiki/Git) je softver za kontrolu verzije i praćenje promene koda koje se lako mogu videti na platformama za hostovanje riznica. Koristiće prvenstveno za kloniranje i apdejtovanje riznice sa komandom *pull*, više o tome u sekciji kloniranje riznice. 

#### Windows:

1. Otvori *Powershell* kao administrator desnim klikom na start pa klikom na *Windows Powershell (Admin)* i nalepi sledeći tekst: <br>
`Set-ExecutionPolicy RemoteSigned -scope CurrentUser` <br> i stisni enter, kada te upita nešto ukucaj `y` i stisni enter.
2. Ugasi *Powershell* i pokreni ga opet na prethodni način ali sada ne kao administrator.
3. Instaliraj [scoop](https://scoop.sh) nalepljivanjem sledećeg teksta u *Powershell*: <br>
`Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')`
4. Konačno, instaliraj Git sa: <br>
`scoop install git`

#### Linux:
Ubuntu, Mint i Debian bazirani distroi: `sudo apt install git` <br>
Arch bazirani distroi: `sudo pacman -S git` <br>
RHEL bazirani distroi: `sudo dnf install git`

### Kloniranje riznice

#### Windows: 

1. Otvori *Powershell* u proizvoljnom folderu (može i na desktop-u) tako što držiš taster *shift* i klikneš desni klikni pa klikneš na *Open Powershell window here*
2. Kloniraj riznicu sa: <br>
`git clone https://codeberg.org/nikolal/VKMKE.git`

#### Linux:

Promeni radni folder u terminalu gde želiš i nalepi ovu komandu:<br> 
`git clone https://codeberg.org/nikolal/VKMKE.git`


### Pokretanje skripte

#### Windows:

1. Otvori *Powershell* u folderu *VKMKE* tako što držiš taster *shift* i klikom na desni klik na prazno mesto u folderu pa klikom na *Open Powershell window here*
2. Pokrećeš skriptu sa: <br>
`python .\zad1.py`

Ukoliko vidiš ispis teksta "Numericka ili simbolicka vrednost..." znači da je skipta uspešno pokrenuta.

**Svaku novu promenu u kodu možeš da dobiješ tako što u Powershell-u u folderu VKMKE ukucaš: `git pull` <br>
Na taj način dobijaš apdejtovani kod, pre pokretanja skripti možeš to uvek da uradiš da bi bio siguran da pokrećeš najnoviju verziju skripte.**

#### Linux:

U terminalu u radnom folderu nalepi: `python3 zad1.py`<br>
Apdejtuješ sa: `git pull` u folderu VKMKE

### Kontakt:

Sve sugestije su dobrodošle na: *nikola@lakic.one*
