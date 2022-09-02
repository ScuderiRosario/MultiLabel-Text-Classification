{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "bc8617de",
   "metadata": {},
   "source": [
    "# Multi-Label Text Classification\n",
    "###### Rosario Scuderi"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84dc9efc",
   "metadata": {},
   "source": [
    "![Titolo](https://wallpapercave.com/wp/wp5342493.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0f23eb3",
   "metadata": {},
   "source": [
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:light-blue; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>Introduzione</center></h2>\n",
    "\n",
    "    \n",
    "\n",
    "L'obbiettivo di questo notebook è mostrare l'implementazione di un **classificatore multi-label** che sia in grado di associare uno o più generi ad una determinata serie anime, utilizzando le rispettive **sinossi**.\n",
    "\n",
    "In primis, si eseguirà un processo di **Data Cleaning** per testare la qualità dei dati e, qualora fosse necessario, modificarli in modo da evitare errori e renderli migliori.\n",
    "\n",
    "In un secondo momento, dopo aver effettuato una breve **analisi**, si testeranno diversi classificatori e tra questi verrà effettuato un **confronto**.\n",
    "\n",
    "In generale si effettueranno i seguenti passaggi:\n",
    "   \n",
    "* [**Presentazione del dataset**](#1)\n",
    "* [**Data Cleaning e riorganizzazione del dataset**](#2)\n",
    "* [**Text Processing (NLP)**](#3)\n",
    "* [**Analisi**](#4)\n",
    "* [**Implementazione e confronto  dei classificatori**](#5)\n",
    "* [**Conclusione e considerazioni finali**](#6)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bed4363",
   "metadata": {},
   "source": [
    "<a id=\"1\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:light-blue; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>1. Presentazione del dataset</center></h2>\n",
    "    \n",
    "\n",
    "\n",
    "I dati provengono dal databases del sito **[My animelist](https://myanimelist.net/)** e includono informazioni sulle varie serie (generi e sinossi); Inoltre, sono stati estratti tramite processi di scraping dal sito stesso.\n",
    "\n",
    "![logo](https://upload.wikimedia.org/wikipedia/commons/7/7a/MyAnimeList_Logo.png)\n",
    "\n",
    "Per informazioni aggiuntive: **[Info](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94738c02",
   "metadata": {},
   "source": [
    "#### 1.1 - Importazione del dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1f9a0a35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAL_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Score</th>\n",
       "      <th>Genres</th>\n",
       "      <th>sypnopsis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Cowboy Bebop</td>\n",
       "      <td>8.78</td>\n",
       "      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>\n",
       "      <td>In the year 2071, humanity has colonized sever...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>Cowboy Bebop: Tengoku no Tobira</td>\n",
       "      <td>8.39</td>\n",
       "      <td>Action, Drama, Mystery, Sci-Fi, Space</td>\n",
       "      <td>other day, another bounty—such is the life of ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>Trigun</td>\n",
       "      <td>8.24</td>\n",
       "      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>\n",
       "      <td>Vash the Stampede is the man with a $$60,000,0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>Witch Hunter Robin</td>\n",
       "      <td>7.27</td>\n",
       "      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>\n",
       "      <td>ches are individuals with special powers like ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>Bouken Ou Beet</td>\n",
       "      <td>6.98</td>\n",
       "      <td>Adventure, Fantasy, Shounen, Supernatural</td>\n",
       "      <td>It is the dark century and the people are suff...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16209</th>\n",
       "      <td>48481</td>\n",
       "      <td>Daomu Biji Zhi Qinling Shen Shu</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Adventure, Mystery, Supernatural</td>\n",
       "      <td>No synopsis information has been added to this...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16210</th>\n",
       "      <td>48483</td>\n",
       "      <td>Mieruko-chan</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Comedy, Horror, Supernatural</td>\n",
       "      <td>ko is a typical high school student whose life...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16211</th>\n",
       "      <td>48488</td>\n",
       "      <td>Higurashi no Naku Koro ni Sotsu</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Mystery, Dementia, Horror, Psychological, Supe...</td>\n",
       "      <td>Sequel to Higurashi no Naku Koro ni Gou .</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16212</th>\n",
       "      <td>48491</td>\n",
       "      <td>Yama no Susume: Next Summit</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Adventure, Slice of Life, Comedy</td>\n",
       "      <td>New Yama no Susume anime.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16213</th>\n",
       "      <td>48492</td>\n",
       "      <td>Scarlet Nexus</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Action, Fantasy</td>\n",
       "      <td>Solar calendar year 2020: grotesque organisms ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>16214 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       MAL_ID                             Name    Score  \\\n",
       "0           1                     Cowboy Bebop     8.78   \n",
       "1           5  Cowboy Bebop: Tengoku no Tobira     8.39   \n",
       "2           6                           Trigun     8.24   \n",
       "3           7               Witch Hunter Robin     7.27   \n",
       "4           8                   Bouken Ou Beet     6.98   \n",
       "...       ...                              ...      ...   \n",
       "16209   48481  Daomu Biji Zhi Qinling Shen Shu  Unknown   \n",
       "16210   48483                     Mieruko-chan  Unknown   \n",
       "16211   48488  Higurashi no Naku Koro ni Sotsu  Unknown   \n",
       "16212   48491      Yama no Susume: Next Summit  Unknown   \n",
       "16213   48492                    Scarlet Nexus  Unknown   \n",
       "\n",
       "                                                  Genres  \\\n",
       "0        Action, Adventure, Comedy, Drama, Sci-Fi, Space   \n",
       "1                  Action, Drama, Mystery, Sci-Fi, Space   \n",
       "2      Action, Sci-Fi, Adventure, Comedy, Drama, Shounen   \n",
       "3      Action, Mystery, Police, Supernatural, Drama, ...   \n",
       "4              Adventure, Fantasy, Shounen, Supernatural   \n",
       "...                                                  ...   \n",
       "16209                   Adventure, Mystery, Supernatural   \n",
       "16210                       Comedy, Horror, Supernatural   \n",
       "16211  Mystery, Dementia, Horror, Psychological, Supe...   \n",
       "16212                   Adventure, Slice of Life, Comedy   \n",
       "16213                                    Action, Fantasy   \n",
       "\n",
       "                                               sypnopsis  \n",
       "0      In the year 2071, humanity has colonized sever...  \n",
       "1      other day, another bounty—such is the life of ...  \n",
       "2      Vash the Stampede is the man with a $$60,000,0...  \n",
       "3      ches are individuals with special powers like ...  \n",
       "4      It is the dark century and the people are suff...  \n",
       "...                                                  ...  \n",
       "16209  No synopsis information has been added to this...  \n",
       "16210  ko is a typical high school student whose life...  \n",
       "16211          Sequel to Higurashi no Naku Koro ni Gou .  \n",
       "16212                          New Yama no Susume anime.  \n",
       "16213  Solar calendar year 2020: grotesque organisms ...  \n",
       "\n",
       "[16214 rows x 5 columns]"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import csv\n",
    "df = pd.read_csv ('anime_with_synopsis.csv')\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c72bd793",
   "metadata": {},
   "source": [
    "Il dataset contiene **16214** records con i **5 attributi** (MAL_ID, Name, Score, Genres, sypnopsis). Molti di questi però non contengono abbastanza informazioni e quindi è necessario effettuare un processo di data cleaning."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16bf7781",
   "metadata": {},
   "source": [
    "<a id=\"2\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:orange; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>2. Data Cleaning e riorganizzazione del dataset</center></h2>\n",
    "    \n",
    "\n",
    "\n",
    "![dc](https://i.ytimg.com/vi/QStMjyxVkqw/maxresdefault.jpg)\n",
    "\n",
    "Per limitare il numero di possibili errori e aumentare le prestazioni dei vari classificatori è utile effettuare un processo di **Data Cleaning** sul dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e82728b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df.drop(['Score'], axis = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6366a1b9",
   "metadata": {},
   "source": [
    "#### 2.1 - Ricerca ed eliminazione dei duplicati"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c6b70a90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAL_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Genres</th>\n",
       "      <th>sypnopsis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>12347</th>\n",
       "      <td>36296</td>\n",
       "      <td>Hinamatsuri</td>\n",
       "      <td>Comedy, Sci-Fi, Seinen, Slice of Life, Superna...</td>\n",
       "      <td>hile reveling in the successful clinching of a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14117</th>\n",
       "      <td>39143</td>\n",
       "      <td>Youkoso! Ecolo Shima</td>\n",
       "      <td>Kids</td>\n",
       "      <td>vironmental education film aimed at children. ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16196</th>\n",
       "      <td>48417</td>\n",
       "      <td>Maou Gakuin no Futekigousha: Shijou Saikyou no...</td>\n",
       "      <td>Magic, Fantasy, School</td>\n",
       "      <td>Second season of Maou Gakuin no Futekigousha: ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16197</th>\n",
       "      <td>48418</td>\n",
       "      <td>Maou Gakuin no Futekigousha: Shijou Saikyou no...</td>\n",
       "      <td>Action, Demons, Magic, Fantasy, School</td>\n",
       "      <td>Second half of Maou Gakuin no Futekigousha: Sh...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       MAL_ID                                               Name  \\\n",
       "12347   36296                                        Hinamatsuri   \n",
       "14117   39143                               Youkoso! Ecolo Shima   \n",
       "16196   48417  Maou Gakuin no Futekigousha: Shijou Saikyou no...   \n",
       "16197   48418  Maou Gakuin no Futekigousha: Shijou Saikyou no...   \n",
       "\n",
       "                                                  Genres  \\\n",
       "12347  Comedy, Sci-Fi, Seinen, Slice of Life, Superna...   \n",
       "14117                                               Kids   \n",
       "16196                             Magic, Fantasy, School   \n",
       "16197             Action, Demons, Magic, Fantasy, School   \n",
       "\n",
       "                                               sypnopsis  \n",
       "12347  hile reveling in the successful clinching of a...  \n",
       "14117  vironmental education film aimed at children. ...  \n",
       "16196  Second season of Maou Gakuin no Futekigousha: ...  \n",
       "16197  Second half of Maou Gakuin no Futekigousha: Sh...  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.duplicated(['Name'])]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c46d4589",
   "metadata": {},
   "source": [
    "All'interno del dataset sono presenti **4 duplciati**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1faa1806",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       16210\n",
       "Name         16210\n",
       "Genres       16210\n",
       "sypnopsis    16202\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=df.drop_duplicates(['Name'])\n",
    "df.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d9f78aa",
   "metadata": {},
   "source": [
    "Dopo l'eliminazione dei duplicati il dataset conta complessivamente **16210** anime ma **8** di questi non hanno un riassunto."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ddd4700",
   "metadata": {},
   "source": [
    "#### 2.2 - Eliminazione degli NA e dei riassunti non validi tra le sinossi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e9e5f9ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       16202\n",
       "Name         16202\n",
       "Genres       16202\n",
       "sypnopsis    16202\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=df.dropna(subset=[\"sypnopsis\"])\n",
    "df.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f52589b8",
   "metadata": {},
   "source": [
    "Osservando ed analizzando i dati in maniera più scrupolosa, è possibile notare che, nonostante siano già stati eliminati i valori NA dalla colonna dei riassunti, alcuni di questi non hanno effettivamente un riassunto ma contengono la seguente frase: \n",
    "\n",
    "**\"No synopsis information has been added to this title. Help improve our database by adding a synopsis here .\"**\n",
    "\n",
    "*Esempio*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6b8376bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAL_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Genres</th>\n",
       "      <th>sypnopsis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>6992</th>\n",
       "      <td>18577</td>\n",
       "      <td>Noobow: Na Kokoro</td>\n",
       "      <td>Kids, Slice of Life</td>\n",
       "      <td>No synopsis information has been added to this...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      MAL_ID               Name               Genres  \\\n",
       "6992   18577  Noobow: Na Kokoro  Kids, Slice of Life   \n",
       "\n",
       "                                              sypnopsis  \n",
       "6992  No synopsis information has been added to this...  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df['Name']=='Noobow: Na Kokoro']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "657f9097",
   "metadata": {},
   "source": [
    "*Eliminazione*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4beb358a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       15493\n",
       "Name         15493\n",
       "Genres       15493\n",
       "sypnopsis    15493\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.drop(df[df.sypnopsis == 'No synopsis information has been added to this title. Help improve our database by adding a synopsis here .'].index)\n",
    "df.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f7e6751",
   "metadata": {},
   "source": [
    "Dal dataset sono stati eliminati **709** record.\n",
    "\n",
    "**PS**. Ora ci sono 15493 elementi.\n",
    "\n",
    "Molte sinossi non riassumono la trama dell'opera ma contengono piccole descrizioni da cui difficilmente è possibile estrarre informazioni utili.\n",
    "\n",
    "*Esempio*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "65eccc9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6994    Hello Kitty version of Heidi , a novel by Joha...\n",
       "Name: sypnopsis, dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['sypnopsis'][df['Name']=='Hello Kitty no Alps no Shoujo Heidi II: Klara to no Deai']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c04c1a81",
   "metadata": {},
   "source": [
    " *Eliminazione dei riassunti più piccoli di 150 caratteri*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cf9ef1eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       10258\n",
       "Name         10258\n",
       "Genres       10258\n",
       "sypnopsis    10258\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=df[(df[\"sypnopsis\"].str.len()>150)]\n",
    "df.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68f0f565",
   "metadata": {},
   "source": [
    "**PS**. Ora ci sono 10258 elementi."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70baf0ec",
   "metadata": {},
   "source": [
    "#### 2.3 - Eliminazione dei records senza un genere"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "34906c91",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       10258\n",
       "Name         10258\n",
       "Genres       10258\n",
       "sypnopsis    10258\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=df.dropna(subset=[\"Genres\"])\n",
    "df.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e993a48",
   "metadata": {},
   "source": [
    "Non ci sono record con l'attributo \"**Genres**\" uguali ad NA ma, analizzando più attentamente, è possibile notare che in alcune t-uple, non è presente un valore **NA**, bensì la parola specifica \"**Unknown**\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c6e662e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       14\n",
       "Name         14\n",
       "Genres       14\n",
       "sypnopsis    14\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[(df[\"Genres\"]==\"Unknown\")].count()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "055b3bcb",
   "metadata": {},
   "source": [
    "Ci sono esattamente **14** elementi senza generi.\n",
    "\n",
    "*Eliminazione*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "8a252ad6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MAL_ID       10244\n",
       "Name         10244\n",
       "Genres       10244\n",
       "sypnopsis    10244\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.drop(df[(df[\"Genres\"]==\"Unknown\")].index)\n",
    "df.count()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0090a03c",
   "metadata": {},
   "source": [
    "**PS**. Ora ci sono 10244 elementi."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abae40c9",
   "metadata": {},
   "source": [
    "#### 2.4 - Ripristino dei riassunti"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "93331963",
   "metadata": {},
   "source": [
    "Alcuni riassunti sono incompleti a causa di problemi dovuti al processo di **scraping** effettuato dal creatore del dataset.\n",
    "\n",
    "*Esempio*\n",
    "\n",
    "\"**other** day, another bounty—such is the life of the often unlucky crew of the Bebop. However, this routine is interrupted when Faye, who is chasing a fairly worthless target on Mars, witnesses an oil tanker suddenly explode, causing mass hysteria...\"\n",
    "\n",
    "**PS**. Manca la parte iniziale.\n",
    "\n",
    "Per ottenere risultati migliori in fase di classificazione, può essere utilie ripristinare i riassunti effettuando lo scraping delle pagine html."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f8e34714",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MAL_ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Genres</th>\n",
       "      <th>sypnopsis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Cowboy Bebop</td>\n",
       "      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>\n",
       "      <td>In the year 2071, humanity has colonized sever...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>Cowboy Bebop: Tengoku no Tobira</td>\n",
       "      <td>Action, Drama, Mystery, Sci-Fi, Space</td>\n",
       "      <td>Another day, another bounty—such is the life o...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>Trigun</td>\n",
       "      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>\n",
       "      <td>Vash the Stampede is the man with a $$60,000,0...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>Witch Hunter Robin</td>\n",
       "      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>\n",
       "      <td>Witches are individuals with special powers li...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>Bouken Ou Beet</td>\n",
       "      <td>Adventure, Fantasy, Shounen, Supernatural</td>\n",
       "      <td>It is the dark century and the people are suff...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   MAL_ID                             Name  \\\n",
       "0       1                     Cowboy Bebop   \n",
       "1       5  Cowboy Bebop: Tengoku no Tobira   \n",
       "2       6                           Trigun   \n",
       "3       7               Witch Hunter Robin   \n",
       "4       8                   Bouken Ou Beet   \n",
       "\n",
       "                                              Genres  \\\n",
       "0    Action, Adventure, Comedy, Drama, Sci-Fi, Space   \n",
       "1              Action, Drama, Mystery, Sci-Fi, Space   \n",
       "2  Action, Sci-Fi, Adventure, Comedy, Drama, Shounen   \n",
       "3  Action, Mystery, Police, Supernatural, Drama, ...   \n",
       "4          Adventure, Fantasy, Shounen, Supernatural   \n",
       "\n",
       "                                           sypnopsis  \n",
       "0  In the year 2071, humanity has colonized sever...  \n",
       "1  Another day, another bounty—such is the life o...  \n",
       "2  Vash the Stampede is the man with a $$60,000,0...  \n",
       "3  Witches are individuals with special powers li...  \n",
       "4  It is the dark century and the people are suff...  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from bs4 import BeautifulSoup\n",
    "import string\n",
    "from zipfile import ZipFile\n",
    "import re\n",
    "\n",
    "def get_description(sum_info):\n",
    "    return sum_info.findAll(\"p\", {\"itemprop\": \"description\"})[0].text\n",
    "\n",
    "def extract_zip(input_zip):\n",
    "    input_zip = ZipFile(input_zip)\n",
    "    return {name: input_zip.read(name) for name in input_zip.namelist()}\n",
    "\n",
    "def get_info_anime(anime_id):\n",
    "    data = extract_zip(f\"AnimeZip2/{anime_id}.zip\")\n",
    "    anime_info = data[\"details.html\"].decode()\n",
    "    \n",
    "    soup = BeautifulSoup(anime_info, \"html.parser\")\n",
    "    description = get_description(soup)\n",
    "    description=description.replace('\\n','')\n",
    "    description=\" \".join(description.split())\n",
    "    description=description.replace('\\n','')\n",
    "\n",
    "    return description\n",
    "\n",
    "df['sypnopsis']=df.apply(lambda x : get_info_anime(x['MAL_ID']),axis=1 )\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37c3bf3d",
   "metadata": {},
   "source": [
    "Alla fine di alcuni riassunti è presente un **riferimento** all'autore o alla fonte da cui è stata estratta la sinossi.\n",
    "\n",
    "*Esempi*\n",
    "\n",
    "- \"...The battle begins. **(Source: ANN)**\"\n",
    "- \"...that will change their lives forever! **(Source: RightStuf)**\"\n",
    "- \"...and regain his reputation as an architect. **[Written by MAL Rewrite]**\"\n",
    "\n",
    "*Eliminazione usando le espressioni regolari.*\n",
    "\n",
    "\n",
    "\n",
    "![re](https://www.ilsoftware.it/public/shots/regular_expression_03_1117.jpg)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "abc41ce0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rm_credit(text):\n",
    "    text = re.sub(\"[\\(\\[].*?[\\)\\]]\", \"\", text) \n",
    "    return text\n",
    "df['sypnopsis']=df.apply(lambda x : rm_credit(x['sypnopsis']),axis=1 )\n",
    "df.to_csv('anime_cleaned.csv') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "619ff384",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>sypnopsis</th>\n",
       "      <th>Genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>In the year 2071, humanity has colonized sever...</td>\n",
       "      <td>Action, Adventure, Comedy, Drama, Sci-Fi, Space</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>Another day, another bounty—such is the life o...</td>\n",
       "      <td>Action, Drama, Mystery, Sci-Fi, Space</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>Vash the Stampede is the man with a $$60,000,0...</td>\n",
       "      <td>Action, Sci-Fi, Adventure, Comedy, Drama, Shounen</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>Witches are individuals with special powers li...</td>\n",
       "      <td>Action, Mystery, Police, Supernatural, Drama, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8</td>\n",
       "      <td>It is the dark century and the people are suff...</td>\n",
       "      <td>Adventure, Fantasy, Shounen, Supernatural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10239</th>\n",
       "      <td>48466</td>\n",
       "      <td>In the year 2061 AD, Japan has lost its sovere...</td>\n",
       "      <td>Action, Mecha</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10240</th>\n",
       "      <td>48470</td>\n",
       "      <td>The stage is Shibuya. When Ryuuhei Oda was in ...</td>\n",
       "      <td>Action, Adventure, Drama, Magic, Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10241</th>\n",
       "      <td>48471</td>\n",
       "      <td>The first astronaut in human history was a vam...</td>\n",
       "      <td>Sci-Fi, Space, Vampire</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10242</th>\n",
       "      <td>48483</td>\n",
       "      <td>Miko is a typical high school student whose li...</td>\n",
       "      <td>Comedy, Horror, Supernatural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10243</th>\n",
       "      <td>48492</td>\n",
       "      <td>Solar calendar year 2020: grotesque organisms ...</td>\n",
       "      <td>Action, Fantasy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10244 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          ID                                          sypnopsis  \\\n",
       "0          1  In the year 2071, humanity has colonized sever...   \n",
       "1          5  Another day, another bounty—such is the life o...   \n",
       "2          6  Vash the Stampede is the man with a $$60,000,0...   \n",
       "3          7  Witches are individuals with special powers li...   \n",
       "4          8  It is the dark century and the people are suff...   \n",
       "...      ...                                                ...   \n",
       "10239  48466  In the year 2061 AD, Japan has lost its sovere...   \n",
       "10240  48470  The stage is Shibuya. When Ryuuhei Oda was in ...   \n",
       "10241  48471  The first astronaut in human history was a vam...   \n",
       "10242  48483  Miko is a typical high school student whose li...   \n",
       "10243  48492  Solar calendar year 2020: grotesque organisms ...   \n",
       "\n",
       "                                                  Genres  \n",
       "0        Action, Adventure, Comedy, Drama, Sci-Fi, Space  \n",
       "1                  Action, Drama, Mystery, Sci-Fi, Space  \n",
       "2      Action, Sci-Fi, Adventure, Comedy, Drama, Shounen  \n",
       "3      Action, Mystery, Police, Supernatural, Drama, ...  \n",
       "4              Adventure, Fantasy, Shounen, Supernatural  \n",
       "...                                                  ...  \n",
       "10239                                      Action, Mecha  \n",
       "10240           Action, Adventure, Drama, Magic, Fantasy  \n",
       "10241                             Sci-Fi, Space, Vampire  \n",
       "10242                       Comedy, Horror, Supernatural  \n",
       "10243                                    Action, Fantasy  \n",
       "\n",
       "[10244 rows x 3 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "anime = pd.read_csv(\"anime_cleaned.csv\")\n",
    "anime =anime.drop(\"Unnamed: 0\",axis=1)\n",
    "anime = anime.drop(\"Name\",axis=1)\n",
    "anime = anime.rename(columns = {'MAL_ID':'ID'})\n",
    "anime = anime[['ID', 'sypnopsis', 'Genres']]\n",
    "anime"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64b58219",
   "metadata": {},
   "source": [
    "Per effettuare la classificazione multi-label, sarà necessario, più avanti, \"**scomporre**\" la lista di generi, in modo da creare un numero di colonne binarie (1 or 0) pari al numero di generi diversi presenti nel dataset.\n",
    "Per farlo utilizzeremo la libreria **sklearn** ma prima è necessario effettuare una conversione delle \"liste\" di generi (da stringa a lista).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fe875b03",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>sypnopsis</th>\n",
       "      <th>Genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>In the year 2071, humanity has colonized sever...</td>\n",
       "      <td>[Action, Adventure, Comedy, Drama, Sci-Fi, Space]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>Another day, another bounty—such is the life o...</td>\n",
       "      <td>[Action, Drama, Mystery, Sci-Fi, Space]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6</td>\n",
       "      <td>Vash the Stampede is the man with a $$60,000,0...</td>\n",
       "      <td>[Action, Sci-Fi, Adventure, Comedy, Drama, Sho...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID                                          sypnopsis  \\\n",
       "0   1  In the year 2071, humanity has colonized sever...   \n",
       "1   5  Another day, another bounty—such is the life o...   \n",
       "2   6  Vash the Stampede is the man with a $$60,000,0...   \n",
       "\n",
       "                                              Genres  \n",
       "0  [Action, Adventure, Comedy, Drama, Sci-Fi, Space]  \n",
       "1            [Action, Drama, Mystery, Sci-Fi, Space]  \n",
       "2  [Action, Sci-Fi, Adventure, Comedy, Drama, Sho...  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#In realtà è necessario eliminare anche gli spazi \n",
    "#la loro presenza non permette all'algoritmo di leggere correttamente le etichette\n",
    "\n",
    "anime['Genres']=anime.apply(lambda x : x['Genres'].replace(\" \",\"\"),axis=1 )\n",
    "anime['Genres']=anime.apply(lambda x : list(x['Genres'].split(\",\")),axis=1 )\n",
    "anime.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3000804c",
   "metadata": {},
   "source": [
    "<a id=\"3\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:blueviolet; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>3. Text Processing (NLP)</center></h2>\n",
    "    \n",
    "    \n",
    "\n",
    "![nlp](https://th.bing.com/th/id/OIP.39A-LtMSZal0Sc9Fe9AIKgHaEK?pid=ImgDet&rs=1)\n",
    "    \n",
    "    \n",
    "Prima di procedere, è necessario **pre-processare** il testo.\n",
    "    \n",
    "Ci sono diversi strumenti a supporto di questa fase; oltre le **espressioni regolari** esistono diverse librerie ed alcune delle più importanti e diffuse sono **spaCy** e **NTLK**.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f2729a5",
   "metadata": {},
   "source": [
    "#### 3.1 - spaCy VS NTLK"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a08b45c2",
   "metadata": {},
   "source": [
    "![spcay](https://spacy.io/static/social_default-1d3b50b1eba4c2b06244425ff0c49570.jpg)\n",
    "![vs](https://cdn5.vectorstock.com/i/thumb-large/92/29/vs-versus-logo-vector-20389229.jpg)\n",
    "![NTLK](https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/OOjs_UI_icon_edit-ltr-progressive.svg/500px-OOjs_UI_icon_edit-ltr-progressive.svg.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1b8fb294",
   "metadata": {},
   "source": [
    "La differenza fondamentale tra **NLTK** e **spaCy** deriva dal modo in cui queste librerie sono state pensate. **NLTK** è essenzialmente una libreria di elaborazione delle **stringhe**, in cui ogni funzione accetta stringhe come input e restituisce una stringa elaborata.\n",
    "Al contrario, **spaCy** adotta un approccio orientato agli oggetti. Ogni funzione restituisce oggetti anziché stringhe o matrici; Inoltre, NLTK restituisce risultati **molto più lenti** rispetto a spaCy.\n",
    "\n",
    "La maggior parte delle fonti su Internet affermano che spaCy supporta solo la lingua inglese, ma questi articoli sono stati scritti alcuni anni fa. Da allora, spaCy è cresciuta fino a supportare molte lingue. Sia spaCy che NLTK supportano inglese, tedesco, francese, spagnolo, portoghese, italiano, olandese e greco.\n",
    "\n",
    "A causa delle grandi dimensioni del dataset si utilizzerà **spaCY**.\n",
    "\n",
    "\n",
    "Estratto da [NTLK VS spaCY](https://www.activestate.com/blog/natural-language-processing-nltk-vs-spacy/) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e65d326f",
   "metadata": {},
   "source": [
    "#### 3.2 - NLP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2c65f7f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy as sp\n",
    "from spacy.lang.en.stop_words import STOP_WORDS\n",
    "nlp = sp.load('en_core_web_sm')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4c65ebf",
   "metadata": {},
   "source": [
    "*Conversione da MAIUSC a MIN*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8e649dbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "anime['sypnopsis']=anime.apply(lambda x : x['sypnopsis'].lower(),axis=1 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "001694b8",
   "metadata": {},
   "source": [
    "*Rimozione Stop-words e punteggiatura*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "abc850af",
   "metadata": {},
   "outputs": [],
   "source": [
    "def stop_and_punct_rm(text):\n",
    "    txt = ''\n",
    "    doc=nlp(text)\n",
    "    for i in doc:\n",
    "        if not (i.is_stop or i.is_punct ):\n",
    "            txt=txt+str(i)+\" \"\n",
    "    return txt\n",
    "\n",
    "anime['sypnopsis']=anime.apply(lambda x : stop_and_punct_rm(x['sypnopsis']),axis=1 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a00876da",
   "metadata": {},
   "source": [
    "*Lemmatization*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "68ab4982",
   "metadata": {},
   "outputs": [],
   "source": [
    "def lemma(text):\n",
    "    txt = ''\n",
    "    doc=nlp(text)\n",
    "    for i in doc:\n",
    "        txt=txt+str(i.lemma_)+\" \"\n",
    "    return txt\n",
    "\n",
    "anime['sypnopsis']=anime.apply(lambda x : lemma(x['sypnopsis']),axis=1 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e10ba8f8",
   "metadata": {},
   "source": [
    "*Rimozione spazi*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "5cb6532b",
   "metadata": {},
   "outputs": [],
   "source": [
    "anime['sypnopsis']=anime.apply(lambda x : x['sypnopsis'].replace(\"\\n\",\" \").strip(),axis=1 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73b85426",
   "metadata": {},
   "source": [
    "*Rimozione numeri*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5655d80d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rm_number(text):\n",
    "    text = re.sub(\"[0-9]\", \"\", text) \n",
    "    return text\n",
    "anime['sypnopsis']=anime.apply(lambda x : rm_number(x['sypnopsis']),axis=1 )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6f41494",
   "metadata": {},
   "source": [
    "*Esempio riassunto post elaborazione*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9549f4d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'edward elric young brilliant alchemist lose year life brother alphonse try resurrect dead mother forbid act human transmutation edward lose brother limbs supreme alchemy skills edward binds alphonse soul large suit armor year later edward promote fullmetal alchemist state embark journey young brother obtain philosopher stone fable mythical object rumor capable amplify alchemist ability leap bound allow override fundamental law alchemy gain alchemist sacrifice equal value edward hope draw military resource find fabled stone restore alphonse body normal elric brother soon discover legendary stone meet eye lead epicenter far dark battle imagine'"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "anime['sypnopsis'][100]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2605ba04",
   "metadata": {},
   "source": [
    "<a id=\"4\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:deeppink; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>4. Analisi</center></h2>\n",
    "    \n",
    "    \n",
    "\n",
    "![eda](https://workhorseconsulting.net/wp-content/uploads/2020/05/DataAnalysisProcess.jpg)\n",
    "    \n",
    "    \n",
    "L'utilizzo di un determinato dataset per l'implementazione di algoritmi di machine learning richiede necessariamente un'**analisi**.\n",
    "    \n",
    "Il processo di analisi aiuta a comprendere meglio le caratteristiche dei dati e come questi sono \"distribuiti\" all'interno del dataset; la conoscenza di queste informazioni è estremamente utile per determinare l'approccio al dataset in fase di implementazione e interpretazione (dei risultati) dei vari algoritmi di machine learning.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3921ba4",
   "metadata": {},
   "source": [
    "Proviamo ad analizzare i generi (dopo averli convertiti in colonne binarie)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "188ca866",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sypnopsis</th>\n",
       "      <th>Action</th>\n",
       "      <th>Adventure</th>\n",
       "      <th>Cars</th>\n",
       "      <th>Comedy</th>\n",
       "      <th>Dementia</th>\n",
       "      <th>Demons</th>\n",
       "      <th>Drama</th>\n",
       "      <th>Ecchi</th>\n",
       "      <th>Fantasy</th>\n",
       "      <th>...</th>\n",
       "      <th>Shounen</th>\n",
       "      <th>ShounenAi</th>\n",
       "      <th>SliceofLife</th>\n",
       "      <th>Space</th>\n",
       "      <th>Sports</th>\n",
       "      <th>SuperPower</th>\n",
       "      <th>Supernatural</th>\n",
       "      <th>Thriller</th>\n",
       "      <th>Vampire</th>\n",
       "      <th>Yaoi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>year  humanity colonize planet moon solar syst...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>day bounty life unlucky crew bebop routine int...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vash stampede man $ $ ,,, bounty head reason m...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 42 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                           sypnopsis  Action  Adventure  Cars  \\\n",
       "0  year  humanity colonize planet moon solar syst...       1          1     0   \n",
       "1  day bounty life unlucky crew bebop routine int...       1          0     0   \n",
       "2  vash stampede man $ $ ,,, bounty head reason m...       1          1     0   \n",
       "\n",
       "   Comedy  Dementia  Demons  Drama  Ecchi  Fantasy  ...  Shounen  ShounenAi  \\\n",
       "0       1         0       0      1      0        0  ...        0          0   \n",
       "1       0         0       0      1      0        0  ...        0          0   \n",
       "2       1         0       0      1      0        0  ...        1          0   \n",
       "\n",
       "   SliceofLife  Space  Sports  SuperPower  Supernatural  Thriller  Vampire  \\\n",
       "0            0      1       0           0             0         0        0   \n",
       "1            0      1       0           0             0         0        0   \n",
       "2            0      0       0           0             0         0        0   \n",
       "\n",
       "   Yaoi  \n",
       "0     0  \n",
       "1     0  \n",
       "2     0  \n",
       "\n",
       "[3 rows x 42 columns]"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import MultiLabelBinarizer\n",
    "from sklearn.preprocessing import StandardScaler  \n",
    "\n",
    "mlb = MultiLabelBinarizer()\n",
    "labels = mlb.fit_transform(anime.Genres)\n",
    "dt = pd.concat([anime[['ID','sypnopsis']], pd.DataFrame(labels)], axis=1)\n",
    "dt.columns = ['ID','sypnopsis'] + list(mlb.classes_)\n",
    "dt = dt.drop(\"ID\",axis=1)\n",
    "dt.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f7ac2f2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "sums=dt.iloc[:, 1:].sum()\n",
    "sums=sums.to_frame()\n",
    "sums=sums.rename(columns = {0:'summ'})\n",
    "number=sums['summ']\n",
    "number.plot.bar()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ea22d17",
   "metadata": {},
   "source": [
    "Notiamo che il numero di elementi che appartengono ai vari generi (**etichette**) è molto vario; Ci sono generi meno presenti rispetto ad altri in maniera molto evidente.\n",
    "\n",
    "Proviamo a bilanciare il dataset eliminando i generi meno rilevanti (molti di questi sono \"sotto generi\"); questo renderà i processi di apprendimento e predizione più semplici, migliorandone le prestazioni."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "7fadc4c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Record rimanenti: 8758\n"
     ]
    }
   ],
   "source": [
    "import warnings\n",
    "import pandas as ps\n",
    "\n",
    "sums=dt.iloc[:, 1:].sum()\n",
    "sums=sums.to_frame()\n",
    "sums=sums.rename(columns = {0:'summ'})\n",
    "sums=sums[sums.summ <= 100]\n",
    "sums=sums.index.tolist()\n",
    "dt=dt = dt.drop(sums, axis=1)\n",
    "\n",
    "warnings.filterwarnings('ignore')  #metodo deprecato\n",
    "sums2=dt.sum(axis=1)\n",
    "sums2=sums2.to_frame()\n",
    "sums2=sums2.rename(columns = {0:'summ'})\n",
    "dt = pd.concat([dt, sums2], axis=1)\n",
    "dt=dt[dt.summ >=2]\n",
    "dt.drop('summ', axis=1, inplace=True)\n",
    "print('Record rimanenti:',dt.shape[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d7d3849d",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "dt.to_csv('final_dataset.csv') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "86afc3e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sypnopsis</th>\n",
       "      <th>Action</th>\n",
       "      <th>Adventure</th>\n",
       "      <th>Comedy</th>\n",
       "      <th>Dementia</th>\n",
       "      <th>Demons</th>\n",
       "      <th>Drama</th>\n",
       "      <th>Ecchi</th>\n",
       "      <th>Fantasy</th>\n",
       "      <th>Game</th>\n",
       "      <th>...</th>\n",
       "      <th>Seinen</th>\n",
       "      <th>Shoujo</th>\n",
       "      <th>Shounen</th>\n",
       "      <th>SliceofLife</th>\n",
       "      <th>Space</th>\n",
       "      <th>Sports</th>\n",
       "      <th>SuperPower</th>\n",
       "      <th>Supernatural</th>\n",
       "      <th>Thriller</th>\n",
       "      <th>Vampire</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>year  humanity colonize planet moon solar syst...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>day bounty life unlucky crew bebop routine int...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vash stampede man $ $ ,,, bounty head reason m...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 37 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                           sypnopsis  Action  Adventure  \\\n",
       "0  year  humanity colonize planet moon solar syst...       1          1   \n",
       "1  day bounty life unlucky crew bebop routine int...       1          0   \n",
       "2  vash stampede man $ $ ,,, bounty head reason m...       1          1   \n",
       "\n",
       "   Comedy  Dementia  Demons  Drama  Ecchi  Fantasy  Game  ...  Seinen  Shoujo  \\\n",
       "0       1         0       0      1      0        0     0  ...       0       0   \n",
       "1       0         0       0      1      0        0     0  ...       0       0   \n",
       "2       1         0       0      1      0        0     0  ...       0       0   \n",
       "\n",
       "   Shounen  SliceofLife  Space  Sports  SuperPower  Supernatural  Thriller  \\\n",
       "0        0            0      1       0           0             0         0   \n",
       "1        0            0      1       0           0             0         0   \n",
       "2        1            0      0       0           0             0         0   \n",
       "\n",
       "   Vampire  \n",
       "0        0  \n",
       "1        0  \n",
       "2        0  \n",
       "\n",
       "[3 rows x 37 columns]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "dt = pd.read_csv ('final_dataset.csv')\n",
    "dt.drop(columns=dt.columns[0], axis=1, inplace=True)\n",
    "dt.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78eafe11",
   "metadata": {},
   "source": [
    "<a id=\"5\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:darkslategray; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>5. Implementazione e confronto  dei classificatori</center></h2>\n",
    "    \n",
    "![ia](https://en.agictech.com/media/1809/machine-learning.jpg)    \n",
    "    \n",
    "In primis, dividiamo il dataset in **train** e **test** set per verificare che i due insiemi abbiano le stesse caratteristiche."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "26144cb8",
   "metadata": {},
   "source": [
    "Estrazione classi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "627a56e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Action</th>\n",
       "      <th>Adventure</th>\n",
       "      <th>Comedy</th>\n",
       "      <th>Dementia</th>\n",
       "      <th>Demons</th>\n",
       "      <th>Drama</th>\n",
       "      <th>Ecchi</th>\n",
       "      <th>Fantasy</th>\n",
       "      <th>Game</th>\n",
       "      <th>Harem</th>\n",
       "      <th>...</th>\n",
       "      <th>Seinen</th>\n",
       "      <th>Shoujo</th>\n",
       "      <th>Shounen</th>\n",
       "      <th>SliceofLife</th>\n",
       "      <th>Space</th>\n",
       "      <th>Sports</th>\n",
       "      <th>SuperPower</th>\n",
       "      <th>Supernatural</th>\n",
       "      <th>Thriller</th>\n",
       "      <th>Vampire</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 36 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Action  Adventure  Comedy  Dementia  Demons  Drama  Ecchi  Fantasy  Game  \\\n",
       "0       1          1       1         0       0      1      0        0     0   \n",
       "1       1          0       0         0       0      1      0        0     0   \n",
       "2       1          1       1         0       0      1      0        0     0   \n",
       "\n",
       "   Harem  ...  Seinen  Shoujo  Shounen  SliceofLife  Space  Sports  \\\n",
       "0      0  ...       0       0        0            0      1       0   \n",
       "1      0  ...       0       0        0            0      1       0   \n",
       "2      0  ...       0       0        1            0      0       0   \n",
       "\n",
       "   SuperPower  Supernatural  Thriller  Vampire  \n",
       "0           0             0         0        0  \n",
       "1           0             0         0        0  \n",
       "2           0             0         0        0  \n",
       "\n",
       "[3 rows x 36 columns]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "#Estraggo i generi\n",
    "y= dt.iloc[: , 1:]\n",
    "y.head(3)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7a15013",
   "metadata": {},
   "source": [
    "Poichè i dati sono testuali, usiamo **TF-IDF** sui riassunti."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "63dd4a66",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<8758x32725 sparse matrix of type '<class 'numpy.float64'>'\n",
       "\twith 363268 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "vectorizer = TfidfVectorizer()\n",
    "X = vectorizer.fit_transform(dt['sypnopsis'])\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0197f484",
   "metadata": {},
   "outputs": [],
   "source": [
    "Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size =0.20)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6d9c397",
   "metadata": {},
   "source": [
    "#### 5.1 - Approcci diversi"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30454192",
   "metadata": {},
   "source": [
    "Per effettuare una classificazione multi-label bisogna effettuare delle \"trasformazioni\" in modo da adattare i vari algoritmia problemi in cui gli item possono appartenere a più classi. In generale ci sono 3 approcci principali:\n",
    "    \n",
    "- **Binary relevance**: Questa tecnica tratta ogni etichetta in modo indipendente e quindi il problema viene trasformato in più problemi di classificazione singola.\n",
    "\n",
    "\n",
    "- **Classifier chains**: In questa tecnica, abbiamo più classificatori collegati in una catena. Si tratta di un processo sequenziale in cui un output di un classificatore viene utilizzato come input del classificatore successivo nella catena.\n",
    "\n",
    "\n",
    "- **Label powerset**: Trasforma il problema in un problema multi-classe. Ogni classificatore multiclasse viene quindi addestrato con combinazioni di etichette univoche presenti nei dati.L'obiettivo è trovare una combinazione di etichette univoche e assegnare loro valori diversi.\n",
    "\n",
    "\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "65aab32a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from skmultilearn.problem_transform import BinaryRelevance\n",
    "from skmultilearn.problem_transform import ClassifierChain\n",
    "from skmultilearn.problem_transform import LabelPowerset\n",
    "\n",
    "import time"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "740caacc",
   "metadata": {},
   "source": [
    "#### 5.2 - Metriche"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a00be21",
   "metadata": {},
   "source": [
    "A causa della natura \"sparsa\" del dataset, è necessario scegliere delle metriche di valutazione adatte;Ad esempio,con i dati presi in considerazione potrebbe non essere conveniente usare l'accuratezza, proprio a causa dei dati estremamente sparpagliati.\n",
    "Le metriche utilizzate, dunque, sono le seguenti:\n",
    "    \n",
    "- **Macro F1**: Il punteggio F1 macro viene calcolato utilizzando la media aritmetica (ovvero la media non ponderata) di tutti i punteggi F1 per classe.\n",
    "\n",
    "\n",
    "- **Micro F1**: La micro media calcola un punteggio F1 medio globale contando le somme dei veri positivi (TP), dei falsi negativi (FN) e dei falsi positivi (FP).\n",
    "\n",
    "\n",
    "- **Hamming**: La perdita di Hamming viene utilizzata per determinare la frazione di previsioni errate di un determinato modello. *Minore è la perdita di hamming, migliore è il nostro modello nel fare previsioni.*\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ef34f5c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import hamming_loss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d186d8e",
   "metadata": {},
   "source": [
    "#### 5.3 - Random Forest\n",
    "\n",
    "Random Forest è un metodo di apprendimento \"ensemble\" che utilizza più alberi decisionali per ottenere risultati migliori. Il risultato finale sarà quello più frequente tra tutti gli esiti dei vari alberi."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "320525a5",
   "metadata": {},
   "source": [
    "![rf](https://miro.medium.com/max/1732/0*x5qFXPNNnbMqMO82)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b28c4adc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RandomForest - BinaryRelevance\n",
      "Macro F1 ->  16.77 %\n",
      "Micro F1 ->  28.21 %\n",
      "Hamming  ->  8.98 %\n",
      "training time taken:  495.0 seconds\n",
      "prediction time taken:  495.0 seconds\n",
      "RandomForest - ClassifierChain\n",
      "Macro F1 ->  14.89 %\n",
      "Micro F1 ->  26.42 %\n",
      "Hamming  ->  9.1 %\n",
      "training time taken:  434.0 seconds\n",
      "prediction time taken:  434.0 seconds\n",
      "RandomForest - LabelPowerset\n",
      "Macro F1 ->  40.88 %\n",
      "Micro F1 ->  47.76 %\n",
      "Hamming  ->  9.99 %\n",
      "training time taken:  506.0 seconds\n",
      "prediction time taken:  506.0 seconds\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "print(\"RandomForest - BinaryRelevance\")\n",
    "start=time.time()\n",
    "classifier = BinaryRelevance(classifier=RandomForestClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"RandomForest - ClassifierChain\")\n",
    "start=time.time()\n",
    "classifier = ClassifierChain(classifier=RandomForestClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"RandomForest - LabelPowerset\")\n",
    "start=time.time()\n",
    "classifier = LabelPowerset(classifier=RandomForestClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fae4f2b3",
   "metadata": {},
   "source": [
    "#### 5.4 - MultinomialNB\n",
    "\n",
    "L'algoritmo si basa sul teorema di Bayes e viene spesso utilizzato per la classificazione di un testo.\n",
    "![rf](https://www.edureka.co/blog/content/ver.1554115042/uploads/2019/04/Naive-Bayes-Derivation-Equation-2-Naive-Bayes-In-R-Edureka.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "2df95600",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MultinomialNB - BinaryRelevance\n",
      "Macro F1 ->  3.31 %\n",
      "Micro F1 ->  9.26 %\n",
      "Hamming  ->  9.94 %\n",
      "training time taken:  1.0 seconds\n",
      "prediction time taken:  1.0 seconds\n",
      "MultinomialNB - ClassifierChain\n",
      "Macro F1 ->  3.98 %\n",
      "Micro F1 ->  11.43 %\n",
      "Hamming  ->  9.83 %\n",
      "training time taken:  0.0 seconds\n",
      "prediction time taken:  0.0 seconds\n",
      "MultinomialNB - LabelPowerset\n",
      "Macro F1 ->  5.19 %\n",
      "Micro F1 ->  21.23 %\n",
      "Hamming  ->  12.57 %\n",
      "training time taken:  5.0 seconds\n",
      "prediction time taken:  5.0 seconds\n"
     ]
    }
   ],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "print(\"MultinomialNB - BinaryRelevance\")\n",
    "start=time.time()\n",
    "classifier = BinaryRelevance(classifier=MultinomialNB(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"MultinomialNB - ClassifierChain\")\n",
    "start=time.time()\n",
    "classifier = ClassifierChain(classifier=MultinomialNB(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"MultinomialNB - LabelPowerset\")\n",
    "start=time.time()\n",
    "classifier = LabelPowerset(classifier=MultinomialNB(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0bddae3",
   "metadata": {},
   "source": [
    "#### 5.5 - LinearSVC\n",
    "L'obiettivo di un LinearSVC (Support Vector Classifier) è quello di adattarsi ai dati forniti, restituendo un iperpiano \"best fit\" che divide o categorizza i dati. Da lì, dopo aver ottenuto l'iperpiano, puoi quindi fornire alcune funzionalità al tuo classificatore per vedere qual è la classe \"prevista\".\n",
    "![rf](https://cmsc426spring2019.github.io/assets/math/ransac3.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "9bda5ce8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LinearSVC - BinaryRelevance\n",
      "Macro F1 ->  40.62 %\n",
      "Micro F1 ->  52.58 %\n",
      "Hamming  ->  7.55 %\n",
      "training time taken:  1.0 seconds\n",
      "prediction time taken:  1.0 seconds\n",
      "LinearSVC - ClassifierChain\n",
      "Macro F1 ->  42.83 %\n",
      "Micro F1 ->  53.76 %\n",
      "Hamming  ->  7.67 %\n",
      "training time taken:  2.0 seconds\n",
      "prediction time taken:  2.0 seconds\n",
      "LinearSVC - LabelPowerset\n",
      "Macro F1 ->  48.57 %\n",
      "Micro F1 ->  55.09 %\n",
      "Hamming  ->  9.07 %\n",
      "training time taken:  20.0 seconds\n",
      "prediction time taken:  20.0 seconds\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import LinearSVC\n",
    "\n",
    "print(\"LinearSVC - BinaryRelevance\")\n",
    "start=time.time()\n",
    "classifier = BinaryRelevance(classifier=LinearSVC(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"LinearSVC - ClassifierChain\")\n",
    "start=time.time()\n",
    "classifier = ClassifierChain(classifier=LinearSVC(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"LinearSVC - LabelPowerset\")\n",
    "start=time.time()\n",
    "classifier = LabelPowerset(classifier=LinearSVC(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "241b24f8",
   "metadata": {},
   "source": [
    "#### 5.6 - KNeighborsClassifier\n",
    "L'algoritmo Knn,scelto a priori un numero di fisso di punti, classifica gli items in base ai dati, rappresentati dai punti, che \"circondano\" il dato da classificare.\n",
    "\n",
    "\n",
    "![rf](https://th.bing.com/th/id/R.47219d9efb93c94f9a7777d5abdc90a7?rik=%2fwhHOh2968gKQw&pid=ImgRaw&r=0&sres=1&sresct=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "7e9a809e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNeighborsClassifier - BinaryRelevance\n",
      "Macro F1 ->  41.26 %\n",
      "Micro F1 ->  47.92 %\n",
      "Hamming  ->  8.49 %\n",
      "training time taken:  15.0 seconds\n",
      "prediction time taken:  15.0 seconds\n",
      "KNeighborsClassifier - ClassifierChain\n",
      "Macro F1 ->  36.24 %\n",
      "Micro F1 ->  45.39 %\n",
      "Hamming  ->  10.09 %\n",
      "training time taken:  17.0 seconds\n",
      "prediction time taken:  17.0 seconds\n",
      "KNeighborsClassifier - LabelPowerset\n",
      "Macro F1 ->  37.02 %\n",
      "Micro F1 ->  44.52 %\n",
      "Hamming  ->  10.75 %\n",
      "training time taken:  1.0 seconds\n",
      "prediction time taken:  1.0 seconds\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "print(\"KNeighborsClassifier - BinaryRelevance\")\n",
    "start=time.time()\n",
    "classifier = BinaryRelevance(classifier=KNeighborsClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"KNeighborsClassifier - ClassifierChain\")\n",
    "start=time.time()\n",
    "classifier = ClassifierChain(classifier=KNeighborsClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n",
    "\n",
    "print(\"KNeighborsClassifier - LabelPowerset\")\n",
    "start=time.time()\n",
    "classifier = LabelPowerset(classifier=KNeighborsClassifier(),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "300cb8c6",
   "metadata": {},
   "source": [
    "#### 5.7 - Rete Neurale - MLP\n",
    "\n",
    "Il Percettrone multistrato (in acronimo MLP dall'inglese Multilayer perceptron) è un modello di rete neurale artificiale che mappa insiemi di dati in ingresso in un insieme di dati in uscita appropriati.\n",
    "\n",
    "È fatta di strati multipli di nodi in un grafo diretto, con ogni strato completamente connesso al successivo. Eccetto che per i nodi in ingresso, ogni nodo è un neurone (elemento elaborante) a cui è associata una funzione di attivazione lineare. Il Percettrone multistrato usa una tecnica di apprendimento supervisionato chiamata backpropagation per l'allenamento della rete.\n",
    "\n",
    "La MLP è una modifica del Percettrone lineare standard e può distinguere i dati che non sono separabili linearmente.\n",
    "\n",
    "*fonte:* Wikipedia\n",
    "![rf](https://www.researchgate.net/profile/Ahmed-Thabit-2/publication/315561192/figure/download/fig4/AS:475189663277059@1490305452719/Figure-4-MLP-neural-network-structure-The-weights-of-the-neural-network-are-updated-in.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a0f550a4",
   "metadata": {},
   "source": [
    "A causa dei tempi di addestramento e predizione estremamente lunghi, in particolare con gli adattamenti **LabelPowerset e ClassifierChain**, verrà utilizzato solo la BinaryRelevance.\n",
    "Dopo una serie di prove,i parametri per l'addestramento della rete che hanno dato i risultati migliori in termini di tempo e validazione sono i seguenti:\n",
    "- **hidden_layer_sizes** = 150 -> rappresenta il numero di neuroni nell'i-esimo strato nascosto.\n",
    "\n",
    "- **max_iter** = 250 -> Numero massimo di iterazioni. Il risolutore itera fino alla convergenza (determinata da 'tol') o questo numero di iterazioni.\n",
    "\n",
    "- **activation** = \"relu\" -> Funzione di attivazione \n",
    "\n",
    "- **early_stopping** = True -> Blocca l'addestramento se il risultato non migliora\n",
    "\n",
    "- **learning_rate** = 'adaptive' -> 'adattivo' mantiene il tasso di apprendimento costante a 'learning_rate_init' finché la perdita di allenamento continua a diminuire. \n",
    "- **solver** = 'lbfgs' -> algoritmo di ottimizzazione\n",
    "\n",
    "- **hidden_layer_sizes**: 150 -> il numero di neuroni nell'i-esimo strato nascosto.\n",
    "\n",
    "*fonte*: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "d59c0885",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MLPClassifier - BinaryRelevance\n",
      "Macro F1 ->  47.65 %\n",
      "Micro F1 ->  55.24 %\n",
      "Hamming  ->  7.81 %\n",
      "training time taken:  1545.0 seconds\n",
      "prediction time taken:  1545.0 seconds\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neural_network import MLPClassifier\n",
    "print(\"MLPClassifier - BinaryRelevance\")\n",
    "start=time.time()\n",
    "classifier = BinaryRelevance(classifier=MLPClassifier(hidden_layer_sizes=150,random_state=100,solver='lbfgs', max_iter=250,activation=\"relu\",early_stopping=True),require_dense = [False, True])\n",
    "classifier.fit(Xtrain, ytrain)\n",
    "predictions = classifier.predict(Xtest)\n",
    "macro_f1 = f1_score(ytest, predictions, average='macro')\n",
    "micro_f1 = f1_score(ytest, predictions, average='micro')\n",
    "hamLoss = hamming_loss(ytest, predictions)\n",
    "print('Macro F1 -> ',round((macro_f1)*100,2),\"%\")\n",
    "print('Micro F1 -> ',round((micro_f1)*100,2),\"%\")\n",
    "print('Hamming  -> ',round((hamLoss)*100,2),\"%\")\n",
    "print('training time taken: ',round(time.time()-start,0),'seconds')\n",
    "print('prediction time taken: ',round(time.time()-start,0),'seconds')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6bf7e996",
   "metadata": {},
   "source": [
    "<a id=\"6\"></a>\n",
    "<div class=\"list-group\" id=\"list-tab\" role=\"tablist\">\n",
    "<h2 class=\"list-group-item list-group-item-action active\" data-toggle=\"list\" style='background:light-blue; border:0; color:white' role=\"tab\" aria-controls=\"home\"><center>Conclusione e considerazioni finali</center></h2>\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "375b9167",
   "metadata": {},
   "source": [
    "Tra tutti gli algoritmi testati, nonostante i tempi di addestramento e predizione molto lunghi e risultati non ottimali, il classificatore MLP risulta essere il migliore, nonostante la differenza con LinearSVC in Labelpowerset sia davvero minima. Il peggiore, invece, è il Multinomial Naive Bayes.\n",
    "\n",
    "Da notare come i risultati, in termini di tempo e metrcihe di validazione, cambi radicalmente modificando il tipo di adattamento utilizzato: considerando il Random Forest si nota una grossa differenza di risultati tra LabelPowerset e gli altri metodi.\n",
    "\n",
    "Inoltre, è possibile notare che generalmente, il metodo di adattamento migliore è LabelPowerset.\n",
    "\n",
    "In conclusione si può affermare che, a causa della scarsa qualità dei dati che risultano essere sparsi e con sinossi non adatte, non è possibile ottenere un classificatore ideale per il dataset in esame."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
