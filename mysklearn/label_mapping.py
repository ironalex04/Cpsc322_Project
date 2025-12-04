# mysklearn/label_mapping.py

# Crime Type Mapping Dictionary
# Groups granular LAPD crime labels into meaningful crime categories
#

crime_mapping = {
    # Theft (non-vehicle)
    "BUNCO, ATTEMPT": "Theft",
    "BUNCO, GRAND THEFT": "Theft",
    "BUNCO, PETTY THEFT": "Theft",
    "SHOPLIFTING - PETTY THEFT ($950 & UNDER)": "Theft",
    "SHOPLIFTING-GRAND THEFT ($950.01 & OVER)": "Theft",
    "SHOPLIFTING - ATTEMPT": "Theft",
    "PICKPOCKET": "Theft",
    "PICKPOCKET, ATTEMPT": "Theft",
    "BIKE - STOLEN": "Theft",
    "BIKE - ATTEMPTED STOLEN": "Theft",
    "THEFT PLAIN - PETTY ($950 & UNDER)": "Theft",
    "THEFT PLAIN - ATTEMPT": "Theft",
    "THEFT, COIN MACHINE - PETTY ($950 & UNDER)": "Theft",
    "THEFT, COIN MACHINE - ATTEMPT": "Theft",
    "THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD": "Theft",
    "THEFT, COIN MACHINE - GRAND ($950.01 & OVER)": "Theft",
    "DISHONEST EMPLOYEE - GRAND THEFT": "Theft",
    "DISHONEST EMPLOYEE - PETTY THEFT": "Theft",
    "DISHONEST EMPLOYEE ATTEMPTED THEFT": "Theft",

    # Fraud
    "THEFT OF IDENTITY": "Fraud",
    "CREDIT CARDS, FRAUD USE ($950 & UNDER": "Fraud",
    "CREDIT CARDS, FRAUD USE ($950.01 & OVER)": "Fraud",
    "COUNTERFEIT": "Fraud",
    "DOCUMENT WORTHLESS ($200 & UNDER)": "Fraud",
    "DOCUMENT WORTHLESS ($200.01 & OVER)": "Fraud",
    "EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)": "Fraud",
    "EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)": "Fraud",
    "DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01": "Fraud",
    "DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER": "Fraud",

    # Robbery
    "DRUNK ROLL": "Robbery",
    "DRUNK ROLL - ATTEMPT": "Robbery",
    "PURSE SNATCHING": "Robbery",
    "PURSE SNATCHING - ATTEMPT": "Robbery",
    "THEFT, PERSON": "Robbery",
    "ROBBERY": "Robbery",
    "ATTEMPTED ROBBERY": "Robbery",
    "EXTORTION": "Robbery",

    # Vehicle Theft
    "VEHICLE - STOLEN": "Vehicle Theft",
    "VEHICLE - ATTEMPT STOLEN": "Vehicle Theft",
    "VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)": "Vehicle Theft",
    "BOAT - STOLEN": "Vehicle Theft",
    "DRIVING WITHOUT OWNER CONSENT (DWOC)": "Vehicle Theft",

    # Burglary
    "BURGLARY": "Burglary",
    "BURGLARY, ATTEMPTED": "Burglary",
    "BURGLARY FROM VEHICLE": "Burglary",
    "BURGLARY FROM VEHICLE, ATTEMPTED": "Burglary",

    # Assault
    "BATTERY - SIMPLE ASSAULT": "Assault",
    "BATTERY POLICE (SIMPLE)": "Assault",
    "BATTERY ON A FIREFIGHTER": "Assault",
    "ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT": "Assault",
    "ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER": "Assault",
    "INTIMATE PARTNER - SIMPLE ASSAULT": "Assault",
    "INTIMATE PARTNER - AGGRAVATED ASSAULT": "Assault",

    # Harassment / Threats
    "CRIMINAL THREATS - NO WEAPON DISPLAYED": "Threats/Harassment",
    "THREATENING PHONE CALLS/LETTERS": "Threats/Harassment",
    "STALKING": "Threats/Harassment",
    "DISTURBING THE PEACE": "Threats/Harassment",
    "LETTERS, LEWD  -  TELEPHONE CALLS, LEWD": "Threats/Harassment",

    # Sex Offense
    "RAPE, FORCIBLE": "Sex Offense",
    "RAPE, ATTEMPTED": "Sex Offense",
    "SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH": "Sex Offense",
    "SEXUAL PENETRATION W/FOREIGN OBJECT": "Sex Offense",
    "LEWD CONDUCT": "Sex Offense",
    "LEWD/LASCIVIOUS ACTS WITH CHILD": "Sex Offense",
    "CHILD PORNOGRAPHY": "Sex Offense",
    "INDECENT EXPOSURE": "Sex Offense",
    "BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM": "Sex Offense",
    "SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ": "Sex Offense",
    "BATTERY WITH SEXUAL CONTACT": "Sex Offense",
    "ORAL COPULATION": "Sex Offense",

    # Trafficking
    "HUMAN TRAFFICKING - COMMERCIAL SEX ACTS": "Trafficking",
    "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE": "Trafficking",
    "PIMPING": "Trafficking",
    "PANDERING": "Trafficking",

    # Homicide & Manslaughter
    "CRIMINAL HOMICIDE": "Homicide",
    "MANSLAUGHTER, NEGLIGENT": "Homicide",

    # Kidnapping
    "KIDNAPPING": "Kidnapping",
    "KIDNAPPING - GRAND ATTEMPT": "Kidnapping",
    "FALSE IMPRISONMENT": "Kidnapping",

    # Vandalism / Property Damage
    "VANDALISM - MISDEAMEANOR ($399 OR UNDER)": "Property Damage",
    "VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)": "Property Damage",
    "TELEPHONE PROPERTY - DAMAGE": "Property Damage",

    # Arson
    "ARSON": "Arson",

    # Weapons
    "BRANDISH WEAPON": "Weapons",
    "WEAPONS POSSESSION/BOMBING": "Weapons",
    "DISCHARGE FIREARMS/SHOTS FIRED": "Weapons",
    "SHOTS FIRED AT INHABITED DWELLING": "Weapons",
    "SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT": "Weapons",

    # Public Order / Other Crimes
    "TRESPASSING": "Other",
    "RESISTING ARREST": "Other",
    "RECKLESS DRIVING": "Other",
    "UNAUTHORIZED COMPUTER ACCESS": "Other",
    "ILLEGAL DUMPING": "Other",
    "DISRUPT SCHOOL": "Other",
    "OTHER MISCELLANEOUS CRIME": "Other"
}
