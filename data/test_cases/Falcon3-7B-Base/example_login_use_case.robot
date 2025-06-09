*** Settings ***
Documentation    This test case verifies the login functionality with correct credentials
Library          Browser
Test Tags        Login Login Form Log In 
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
Login with Correct Credentials
    [Documentation]    This test case verifies the login functionality with correct credentials
    [Tags]    Login Login Form Log In 
    
    New Browser    ${BROWSER}    headless=${HEADLESS}
    New Page    ${BASE_URL}/login
    Type Text    id=username    tomsmith
    Type Text    id=password    SuperSecretPassword!
    Click    css=button[type='submit']
    Wait For Elements State    text=You logged into a secure area!    visible
    Take Screenshot

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
