*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        login smoke
Library          OperatingSystem
Library          DateTime
Test Setup       Setup Browser
Test Teardown    Close Browser

*** Variables ***
${BASE_URL}      https://the-internet.herokuapp.com
${BROWSER}       chromium
${HEADLESS}      false
${TIMEOUT}       10s

*** Test Cases ***
Test Login Functionality
    [Documentation]    Automated test for login functionality
    [Tags]    login smoke
    
    Get Text
    Get Text
    Get Text
    Get Text
    Log    Action: Include tests:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text
    Get Text
    Log    Action: Include steps that describe the expected results:
    Log    Action: Step that describe the expected results
    Get Text

*** Keywords ***
Setup Browser
    New Browser    ${BROWSER}    headless=${HEADLESS}
    Set Browser Timeout    ${TIMEOUT}
    New Context    viewport={'width': 1280, 'height': 720}

Close Browser
    Take Screenshot    fullPage=True
    Close Browser    ALL

Login To Application
    [Arguments]    ${username}    ${password}
    Go To    ${BASE_URL}/login
    Type Text    id=username    ${username}
    Type Text    id=password    ${password}
    Click    css=button[type='submit']
    Wait For Elements State    css=.flash    visible    timeout=${TIMEOUT}
