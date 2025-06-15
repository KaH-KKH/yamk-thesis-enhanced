*** Settings ***
Documentation    Verifies the failure of login process using incorrect credentials.
Library          Browser
Test Tags        login security
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
    [Documentation]    Verifies the failure of login process using incorrect credentials.
    [Tags]    login security
    [Setup]    Run Keywords
    ...    Open Browser [URL=https://the-internet.herokuapp.com] [Title=The Internet]
    ...    Wait Until Text Present [Text=Welcome] [Timeout=10s]
    ...    Click Element With XPath=//a[@href='/login'] [Wait=1s]
    
    Click    text="E"
    Type Text    id="Text"    test value
    Type Text    id="Text"    test value
    Click    text="E"
    Wait For Elements State
    Click    text="E"
    Type Text    id="Text"    test value
    Type Text    id="Text"    test value
    Click    text="E"
    Wait For Elements State
    Type Text    id="Text"    test value
    Type Text    id="Text"    test value
    Click    text="E"
    Wait For Elements State
    Click    text="E"
    Type Text    id="Text"    test value
    Type Text    id="Text"    test value
    Click    text="E"
    Wait For Elements State
    
    [Teardown]    Run Keywords
    ...    Close Browser
    ...    Expected Outcomes:
    ...    The system should display an error message indicating invalid credentials.
    ...    The user should remain on the login page without accessing the secure area.
    ...    ```

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
