Gebruikershandleiding Transformer Trainer & Evaluator
Introductie

Welkom bij de gebruikershandleiding voor de Transformer Trainer en Evaluator. Dit document leidt u stapsgewijs door het installatieproces en het gebruik van de software. Volg de stappen in de aangegeven volgorde om een correcte installatie te garanderen.

Het proces is opgedeeld in de volgende fasen:

    Vereisten: Zorg ervoor dat de benodigde databestanden aanwezig zijn.

    Systeemvoorbereiding: Installatie van de vereiste NVIDIA stuurprogramma's en CUDA Toolkit.

    Projectinstallatie: Klonen van de projectbestanden en installatie van de Python-pakketten.

    Hyperparameter Configuratie: Het instellen van de modelparameters.

    Training & Evaluatie: Het daadwerkelijk trainen en evalueren van het model.

    Volgende Stappen: Aanbevelingen voor toekomstige uitbreidingen.

    Veelgestelde Vragen (FAQ): Antwoorden op veelvoorkomende vragen.

Stap 0: Vereisten

Voordat u begint, is het essentieel dat de juiste databestanden aanwezig zijn in de hoofdmap van het project.

    Data-invoer via CSV: Het systeem is geconfigureerd om data te lezen uit CSV-bestanden. De paden naar deze bestanden worden gespecificeerd in config.yaml onder data.train_file_path en data.eval_file_path.

    Trainings- en Evaluatiedata: U heeft minimaal één CSV-bestand met historische data nodig. Dit bestand kan zowel voor training als voor evaluatie worden gebruikt. In dat geval vult u voor beide paden in config.yaml dezelfde bestandsnaam in. Zie de FAQ voor details over het dataformaat.

Stap 1: Installatie van GPU-stuurprogramma en CUDA Toolkit

Deze eerste stap zorgt ervoor dat uw systeem over de juiste software beschikt om de grafische kaart (GPU) te kunnen benutten. Dit is essentieel voor het trainen van de modellen.

    Zoek het bestand GPUDRIVER_CUDA_INSTALL.bat in de projectmap.

    Klik met de rechtermuisknop op het bestand en selecteer "Als administrator uitvoeren".

    Het script controleert of de benodigde NVIDIA-software reeds is geïnstalleerd en installeert deze indien nodig.

    Wanneer het script is voltooid, wordt een herstart van uw systeem ten zeerste aanbevolen.

Stap 2: Projectinstallatie en Python Omgeving

Nadat de systeemvereisten zijn geïnstalleerd en het systeem opnieuw is opgestart, installeert u de projectspecifieke software.

    Zoek het bestand installer.bat in de projectmap en dubbelklik hierop.

    Het script installeert Python, Git, downloadt de projectbestanden en installeert alle Python-pakketten.

    Wacht tot het script de melding "SETUP COMPLETE!" toont.

Stap 3: Hyperparameters Configureren

Alle instellingen beheert u centraal in het config.yaml bestand. De belangrijkste taak is het vinden van de optimale balans tussen VRAM-gebruik, snelheid en nauwkeurigheid.
Aanbevolen Werkwijze

    Open config.yaml.

    Monitor uw GPU: Voer Run_GPU_Monitor.bat uit.

    Start de training: Voer Run_Trainer.bat uit.

    Optimaliseer batch_size: Zoek de hoogste batch_size die werkt zonder "Out of Memory" fouten.

    Optimaliseer Modelcomplexiteit: Verlaag eventueel hidden_size of lookback_window als de training te traag is.

Stap 4: Gebruik van de Trainer en Evaluator
Het model trainen

Voer het Run_Trainer.bat bestand uit. Dit script start de training op basis van de instellingen in config.yaml.

    Epoch: Eén volledige trainingscyclus door de dataset.

    Voortgangsbalk: Toont de val_loss (prestatiescore, lager is beter) en de resterende tijd per epoch.

    Early Stopping & Model Opslaan: Het script slaat na elke epoch die een verbetering in val_loss laat zien, automatisch het model op als het nieuwe 'beste' model. De training stopt vanzelf als er een bepaald aantal epochs geen verbetering optreedt (ingesteld via patience in de config). U kunt het trainingsproces echter op elk moment handmatig stoppen (met Ctrl+C in het terminalvenster); het laatst opgeslagen beste model blijft altijd bewaard in de model_checkpoints map.

Het model evalueren

Het run_eval.bat script vergelijkt voorspellingen met bekende, werkelijke data om de nauwkeurigheid van het model te meten. De resultaten worden weggeschreven naar een Excel-rapport.
Stap 5: Volgende Stappen en Aanbevelingen
Aanbeveling 1: Een apart script voor Toekomstvoorspellingen

Ontwikkel een Predict.py script dat het getrainde model gebruikt om voorspellingen voor toekomstige, onbekende data te genereren.
Aanbeveling 2: Directe Databaseconnectie

Voor productie, pas de datalader aan om data direct uit een Azure Datawarehouse te lezen in plaats van uit CSV-bestanden.
Aanbeveling 3: Integratie van de Feature Enhancer

Een toekomstig Predict.py script moet het date_deriv_feat_enhancer.py script gebruiken om voor de te voorspellen datums exact dezelfde features te genereren waarop het model getraind is.
Stap 6: Veelgestelde Vragen (FAQ)
V: Wat is de exacte structuur die mijn CSV-bestanden moeten hebben?

A: Uw CSV-bestand moet de volgende structuur hebben:

    Verplichte kolommen:

        Een datumkolom (standaard Datum). Het specifieke datumformaat is flexibel, zolang het consistent is.

        Een groepsindicator-kolom (standaard Naam) die de entiteiten groepeert die u wilt voorspellen (bijv. klantnamen, product-ID's).

    Doelkolommen (Target Variables):

        Dit zijn de kolommen die u wilt voorspellen. De namen van deze kolommen zijn volledig instelbaar in config.yaml onder data.target_columns.

        De logica achter de kolomnamen in het voorbeeld (zoals OrderDag, OrderUur_0, Picktime_23) komt voort uit een specifieke SQL-query die de data voorbereidt. U hoeft deze structuur niet exact te volgen, zolang uw doelkolommen numeriek zijn.

    Voorbeeld Data-generatie (SQL):
    De onderstaande query toont hoe de voorbeelddata is gegenereerd. Het aggregeert data per uur en pivoteert dit naar kolommen. Dit is slechts een voorbeeld; uw eigen data kan een andere structuur hebben.

    WITH DailyAggregates AS (
        -- Aggregeert orders en picktijden per klant, per dag en per uur.
        SELECT
            p.LicenseeKey, od.[Date], ot.[Hour],
            COUNT(p.PickLineId) AS HourlyOrders,
            SUM(CAST(p.MillisecondsStartToEndPick AS BIGINT)) AS HourlyPickTime
        FROM FactPicking p
        JOIN DimOrderCreationDate od ON p.LineCreationDateKey = od.DateKey
        JOIN DimOrderCreationTime ot ON p.LineCreationTimeKey = ot.TimeKey
        WHERE p.PickLineId IS NOT NULL AND p.MillisecondsStartToEndPick > 0
        GROUP BY p.LicenseeKey, od.[Date], ot.[Hour]
    ),
    PivotedData AS (
        -- Pivoteert de uurdata naar aparte kolommen.
        SELECT
            da.LicenseeKey, da.[Date],
            SUM(da.HourlyOrders) AS OrderDag,
            -- Pivot voor orders per uur
            SUM(CASE WHEN da.[Hour] = 0 THEN da.HourlyOrders ELSE 0 END) AS OrderUur_0,
            -- ... (herhaald voor uren 1 t/m 23)
            -- Pivot voor picktijd per uur
            SUM(CASE WHEN da.[Hour] = 0 THEN da.HourlyPickTime ELSE 0 END) AS Picktime_0
            -- ... (herhaald voor uren 1 t/m 23)
        FROM DailyAggregates da
        GROUP BY da.LicenseeKey, da.[Date]
    )
    SELECT
        ct.Name AS Naam, pd.[Date] AS Datum, pd.OrderDag,
        pd.OrderUur_0, pd.OrderUur_1, -- ... etc.
        pd.Picktime_0, pd.Picktime_1 -- ... etc.
    FROM PivotedData pd
    JOIN (SELECT DISTINCT LicenseeKey, Name FROM DimLicensee) ct ON pd.LicenseeKey = ct.LicenseeKey
    WHERE ct.Name IS NOT NULL AND ct.Name <> ''
    ORDER BY ct.Name, pd.[Date];

    Belangrijk: U hoeft geen van de datum afgeleide features (zoals dag van de week) handmatig aan uw CSV toe te voegen. Het script doet dit automatisch via de date_deriv_feat_enhancer.

V: Hoeveel historische data heb ik minimaal nodig?

A: Het script filtert automatisch groepen (bijv. klanten) uit die minder datapunten hebben dan de lookback_window die is ingesteld in config.yaml. Zorg er dus voor dat elke groep die u wilt voorspellen ten minste lookback_window dagen aan data heeft.
V: Wat zijn goede startwaarden voor de hyperparameters?

A: De standaardwaarden in config.yaml zijn geoptimaliseerd voor een T4 GPU op een Azure VM, wat een veelvoorkomende use case is. Dit is een uitstekend startpunt. Als u een andere GPU heeft, volgt u de optimalisatiewerkwijze in Stap 3.
V: Hoe weet ik of mijn val_loss 'goed' is?

A: val_loss is een relatieve maatstaf; lager is altijd beter. De absolute waarde is minder belangrijk dan de trend. De beste manier om voortgang te meten, is door de val_loss te vergelijken met die van eerdere trainingsruns. Als de val_loss daalt, leert het model. Ter referentie, een goed presterend model behaalde een val_loss van circa 35,407,444.00, wat resulteerde in een bestandsnaam die eindigde op ...val_loss=35407444.00.ckpt.
V: Wat betekenen de andere instellingen in config.yaml?

A:

    learning_rate: Bepaalt de grootte van de stappen die het model neemt tijdens het leren. Een kleinere waarde leidt tot langzamer maar potentieel nauwkeuriger leren.

    dropout: Een techniek die overfitting tegengaat door tijdens de training willekeurig een deel van de neuronen uit te schakelen. Dit dwingt het model om robuustere patronen te leren.

    optimizer: Het algoritme dat wordt gebruikt om het model te verbeteren. "AdamW" is een moderne en betrouwbare standaardkeuze.

V: Hoe interpreteer ik de output in het Excel-rapport?

A:

    MAE (Mean Absolute Error): De gemiddelde absolute afwijking tussen voorspelde en werkelijke waarden, uitgedrukt in dezelfde eenheid als uw doelvariabele (bijv. "gemiddeld zaten we er 2.5 orders naast").

    MAPE (Mean Absolute Percentage Error): De gemiddelde procentuele afwijking. Nuttig om de fout relatief te zien (bijv. "gemiddeld zaten we er 10% naast").

    R2 Score: Geeft aan welk deel van de variabiliteit in de data door het model wordt verklaard. Een score van 1 is perfect; 0 betekent dat het model niet beter is dan simpelweg het gemiddelde voorspellen.

    Per-klant-tabbladen: Deze bladen tonen een directe vergelijking van de actual (werkelijke) en predicted (voorspelde) waarden voor elke klant, per voorspelde variabele en datum. Dit is ideaal voor een gedetailleerde analyse.

V: Waar wordt het getrainde model opgeslagen?

A: In de map model_checkpoints in de hoofdmap van het project. Deze map wordt automatisch aangemaakt als deze niet bestaat. Het best presterende model (laagste val_loss) herkent u aan "best-model" in de bestandsnaam.
V: De training stopt met een onbekende fout. Wat nu?

A: De volledige foutmelding is zichtbaar in het terminalvenster waarin het Run_Trainer.bat script draait. Kopieer deze foutmelding en plak deze in een LLM zoals Gemini of ChatGPT. Deze tools kunnen vaak de oorzaak van het probleem uitleggen en een oplossing voorstellen.
Probleemoplossing

    Administratorrechten: Voer GPUDRIVER_CUDA_INSTALL.bat altijd als administrator uit.

    Out of Memory: Verlaag de batch_size in config.yaml.

    Internetverbinding: Een stabiele verbinding is nodig voor de installatie.

    Installatielogs: Bij fouten, raadpleeg C:\ProgramData\chocolatey\logs\chocolatey.log.
