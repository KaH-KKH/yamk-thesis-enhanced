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
    [Setup]    Run Keywords
    ...    Title Should Be      Log in
    ...    Test ***
    ...    [Documentation]    Test: Enter valid credentials and click Login
    ...    InputText            username           ${USERNAME}
    ...    InputText            password         ${PASSWORD}
    ...    ClickButton           login_button
    ...    [Documentation]    Test: Verify success message is displayed
    ...    WaitUntilPageContains  Welcome to the Secure Area, tomsmith!
    ...    [Documentation]    Test: Verify user is in the secure area
    ...    Title Should Be      Secure Area: To Do List
    ...    Teardown ***
    
    New Browser    ${BROWSER}    headless=${HEADLESS}
    New Page    ${BASE_URL}/login
    Type Text    id=username    tomsmith
    Type Text    id=password    SuperSecretPassword!
    Click    css=button[type='submit']
    Wait For Elements State    text=You logged into a secure area!    visible
    Take Screenshot
    
    [Teardown]    Run Keywords
    ...    GoToUrl              /login
    ...    Title Should Be      Log in
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
