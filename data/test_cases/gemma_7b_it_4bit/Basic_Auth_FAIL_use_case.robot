*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        smoke regression
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
User cancels login and opens a different page
    [Documentation]    Automated test for login functionality
    [Tags]    smoke regression
    [Setup]    Run Keywords
    ...    Open the internet website "the-internet.herokuapp.com/basic_auth"
    
    Log    Action: The website prompts the user to sign in and displays a message "This site is asking you to sign in".
    Log    Action: Press the "Cancel" button.
    Log    Action: The browser opens a new page with the text "Not authorized".
    
    [Teardown]    Run Keywords
    ...    Close the browser
    ...    ```
    ...    Please note that this is an example and may need to be modified based on the specific requirements of the use case.**

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
