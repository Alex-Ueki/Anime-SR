""" Tests for Modules/misc.py """

import Modules.misc as misc

# Testing utility strings

_T1 = 'Testing 1 2 3. This is a test of the emergency testing system.'
_T2 = 'Stay tuned for bad news about your code'

_T1_N = _T1 + '\n'
_T2_N = _T2 + '\n'

_F1 = 'Format String {}'
_F1_N = _F1 + '\n'

def test_set_docstring():
    """ Test for misc.set_docstring """

    misc.set_docstring(_T1)
    assert misc._DOCSTRING == _T1

def test_clear_screen():
    """ No test for misc.clear_screen because clearing screen is
        done via system call
    """

    pass

def test_printlog(capsys):
    """ Test for misc.printlog """

    # Unformatted log print

    out, err = capsys.readouterr()
    misc.printlog(_T1)
    out, err = capsys.readouterr()

    assert err == ''

    outs = out.split(': ')
    assert len(outs) == 2
    assert outs[1] == _T1_N

    # Formatted log print

    out, err = capsys.readouterr()
    misc.printlog(_T1, _T2)
    out, err = capsys.readouterr()

    outs = out.split(': ')
    assert len(outs) == 2
    assert outs[1] == _T1 + ' ' + _T2_N

def test_setup_default():
    """ No test because it creates folders """

    pass

def test_oops(capsys):
    """ Test of misc.oops """

    # Valid parameter, no prior errors

    out, err = capsys.readouterr()
    result = misc.oops(False, 1 == 0, _T1)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert not result

    # Valid parameter, prior errors

    out, err = capsys.readouterr()
    result = misc.oops(True, 1 == 0, _T1)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result

    # Invalid parameter, no prior errors

    out, err = capsys.readouterr()
    result = misc.oops(False, 1 == 1, _T1)
    out, err = capsys.readouterr()

    assert out == 'Error: ' + _T1_N
    assert err == ''
    assert result

    # Invalid parameter, prior errors

    out, err = capsys.readouterr()
    result = misc.oops(True, 1 == 1, _T1)
    out, err = capsys.readouterr()

    assert out == 'Error: ' + _T1_N
    assert err == ''
    assert result

    # Invalid parameter, prior errors, formatted error string

    out, err = capsys.readouterr()
    result = misc.oops(True, 1 == 1, _F1, _T1)
    out, err = capsys.readouterr()

    assert out == 'Error: ' + _F1_N.format(_T1)
    assert err == ''
    assert result

def test_validate():
    """ Test of misc.validate. Only need to check results because
        all the rest is handled by misc.oops()
    """

    # Valid parameter, no prior errors

    result = misc.validate(False, 123, 1 == 0, _T1)
    assert result == (123, False)

    # Valid parameter, prior errors

    result = misc.validate(True, 'abc', 1 == 0, _T1)
    assert result == ('abc', True)

    # Invalid parameter, no prior errors

    result = misc.validate(False, (123, 'abc'), 1 == 1, _T1)
    assert result == ((123, 'abc'), True)

    # Invalid parameter, prior errors

    result = misc.validate(True, 123.456, 1 == 1, _T1)
    assert result == (123.456, True)

def test_terminate(capsys):
    """ Test of misc.terminate """

    # Do not terminate, not verbose

    out, err = capsys.readouterr()
    misc.terminate(False, False)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''

    # Do not terminate, verbose

    out, err = capsys.readouterr()
    misc.terminate(False, True)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''

    # terminate() is test aware, and will not exit if we are testing

    # Terminate, not verbose

    out, err = capsys.readouterr()
    misc.terminate(True, False)
    out, err = capsys.readouterr()

    assert out == '\n'
    assert err == ''

    # Terminate, verbose

    out, err = capsys.readouterr()
    misc.terminate(True, True)
    out, err = capsys.readouterr()

    assert out == _T1 + '\n\n'
    assert err == ''


def test_opcheck(capsys):
    """ Test of misc.capsys """

    options = ('parameter', str, lambda x: x == _T1, _F1)

    # Valid new parameter, no prior errors

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', '456', False)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == ('456', False)

    # Valid new parameter, prior errors

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', '456', True)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == ('456', True)

    # Invalid new parameter, no prior errors

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', _T1, False)
    out, err = capsys.readouterr()

    assert out == 'Error: ' + _F1_N.format(_T1)
    assert err == ''
    assert result == ('123', True)

    # Type conversion tests

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', 456, False)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == ('456', False)

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', True, False)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == ('True', False)

    out, err = capsys.readouterr()
    result = misc.opcheck(options, '123', 456.789, False)
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == ('456.789', False)

def test_parse_options(capsys):
    """ Test parse_options parsing """

    opcodes = {
        'str': ('str', str, lambda x: x == 'abc', ''),
        'int': ('int', int, lambda x: x <= 0, 'Integer invalid ({})'),
        'float': ('float', float, lambda x: False, ''),
        'bool': ('bool', bool, lambda x: not isinstance(x, bool), 'Boolean invalid.'),
        'path1': ('path1_path', str, lambda x: False, ''),
        'path2': ('path2_path', str, lambda x: False, ''),
    }

    # Basic parsing

    out, err = capsys.readouterr()
    result = misc.parse_options(opcodes, ['str=123', 'int=456'])
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == {'str': '123', 'int': 456, 'paths': {}}

    # Bad parameter format

    out, err = capsys.readouterr()
    result = misc.parse_options(opcodes, ['derf', 'str=123', 'int=456'])
    out, err = capsys.readouterr()

    assert out == 'Error: Invalid option (derf)\n' + _T1 + '\n\n'
    assert err == ''
    assert result == {'str': '123', 'int': 456, 'paths': {}}

    # Unknown parameter

    out, err = capsys.readouterr()
    result = misc.parse_options(opcodes, ['derf=abc', 'str=123', 'int=456'])
    out, err = capsys.readouterr()

    assert out == 'Error: Unknown option (derf)\n' + _T1 + '\n\n'
    assert err == ''
    assert result == {'str': '123', 'int': 456, 'paths': {}}

    # Ambiguous parameter

    out, err = capsys.readouterr()
    result = misc.parse_options(opcodes, ['path=abc', 'str=123', 'int=456'])
    out, err = capsys.readouterr()

    assert out == 'Error: Ambiguous option (path)\n' + _T1 + '\n\n'
    assert err == ''
    assert result == {'str': '123', 'int': 456, 'paths': {}}

    # Path conversion

    out, err = capsys.readouterr()
    result = misc.parse_options(opcodes, ['path1=abc', 'path2=def'])
    out, err = capsys.readouterr()

    assert out == ''
    assert err == ''
    assert result == {'paths': {'path1': 'abc', 'path2': 'def'}}
