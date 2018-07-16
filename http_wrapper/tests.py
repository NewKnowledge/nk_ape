from app import get_predictions


def test_ape():

    print('testing ape')
    # input_words = 'gorilla,chimp,orangutan,gibbon,human'
    input_words = ['gorilla', 'chimp', 'orangutan', 'gibbon', 'human']
    result = get_predictions(input_words)
    print(result)
    assert result
