import numpy as np

def valeur_carte(carte: str) -> int:
    """
    Return the point value of a Blackjack card.
    - Face cards are worth 10.
    - An Ace is worth 11 (will be adjusted down to 1 if needed when computing the score).
    """
    if carte in ['J', 'Q', 'K']:
        return 10
    elif carte == 'A':
        return 11
    else:
        return int(carte)

def pioche_jeu(n: int = 1) -> list[str]:
    """
    Draw n random cards from a standard Blackjack shoe (with replacement).
    """
    cartes = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    return list(np.random.choice(cartes, size=n, replace=True))

def calcul_score_main(main: list[str]) -> int:
    """
    Compute the score of a hand, handling Aces so they can count as 1 if necessary.
    """
    score = sum([valeur_carte(c) for c in main])
    nb_as = main.count('A')
    # If the score exceeds 21 and there are Aces, each Ace can be downgraded from 11 to 1 until we are back to 21 or less.
    while score > 21 and nb_as > 0:
        score -= 10
        nb_as -= 1
    return score

def joue_coup_joueur(main: list[str]) -> list[str]:
    """
    Very simple strategy:
    - While the score < 17, hit; otherwise stand.
    """
    while calcul_score_main(main) < 17:
        main += pioche_jeu()
    return main

def joue_coup_croupier(main: list[str]) -> list[str]:
    """
    Dealer strategy: hit until reaching a score >= 17 (classic rule).
    """
    while calcul_score_main(main) < 17:
        main += pioche_jeu()
    return main

def simule_partie_blackjack() -> int:
    """
    Simulate a Blackjack hand between a player and the dealer.
    Returns:
        1 if the player wins,
        0 for a push (tie),
        -1 if the player loses.
    """
    main_joueur = pioche_jeu(2)
    main_croupier = pioche_jeu(2)

    # Player's turn
    main_joueur = joue_coup_joueur(main_joueur)
    # If the player busts (> 21), he immediately loses
    if calcul_score_main(main_joueur) > 21:
        return -1

    # Dealer's turn
    main_croupier = joue_coup_croupier(main_croupier)
    score_joueur = calcul_score_main(main_joueur)
    score_croupier = calcul_score_main(main_croupier)

    # Outcome
    if score_croupier > 21:
        return 1  # dealer busts
    elif score_joueur > score_croupier:
        return 1  # player wins
    elif score_joueur == score_croupier:
        return 0  # push (tie)
    else:
        return -1  # dealer wins

def monte_carlo_blackjack(nb_simulations: int = 100_000) -> None:
    """
    Simulate nb_simulations Blackjack hands and print win statistics.
    """
    # Use numpy to store results efficiently
    resultats = np.zeros(nb_simulations, dtype=int)
    for i in range(nb_simulations):
        # Store -1, 0, or 1 depending on each game's outcome
        resultats[i] = simule_partie_blackjack()

    # Probability calculations:
    #
    # The Monte Carlo estimate of the probability of event E is
    #   p(E) = (number of observed successes for E) / (total number of simulations)
    proba_gagne = np.sum(resultats == 1) / nb_simulations
    proba_perdu = np.sum(resultats == -1) / nb_simulations
    proba_egalite = np.sum(resultats == 0) / nb_simulations

    print(f"Number of simulations          : {nb_simulations}")
    print(f"Probability of winning (player): {proba_gagne:.4f}")
    print(f"Probability of losing  (player): {proba_perdu:.4f}")
    print(f"Probability of a push          : {proba_egalite:.4f}")

if __name__ == "__main__":
    # Run 100,000 Blackjack simulations
    monte_carlo_blackjack(100_000)