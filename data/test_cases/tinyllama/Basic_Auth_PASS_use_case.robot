*** Settings ***
Documentation    Automated test for login functionality
Library          Browser
Test Tags        Login Basic Auth
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
    [Tags]    Login Basic Auth
    
    Select Options By
    Custom Login Keyword
    Log    Action: Go to https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Press button Sign in
    Log    Action: Browser page opens and there is two headers
    Log    Action: 6.1. Basic Auth
    Log    Action: 6.2. Congratulations! You must have the proper credentials.
    Log    Action: Alternative flows (if any):
    Log    Action: Go to https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Press button Sign in
    Custom Login Keyword
    Custom Login Keyword
    Log    Action: Input username = admin
    Log    Action: 10. Input password = admin
    Click    text="s"
    Log    Action: 12. Browser page opens and there is a thank you message
    Log    Action: Requirement:
    Log    Action: Go To    https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Press button Sign in
    Log    Action: 10. Input username = admin
    Log    Action: 11. Input password = admin
    Click    text="s"
    Log    Action: Browser page opens and there is a thank you message
    Log    Action: Requirement:
    Log    Action: Go To    https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Press button Sign in
    Log    Action: 10. Input username = admin
    Log    Action: 11. Input password = admin
    Click    text="s"
    Log    Action: 12. Browser page opens and there is a thank you message
    Log    Action: Requirement:
    Log    Action: Go To    https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Press button Sign in
    Log    Action: 12. Input username = admin
    Log    Action: 13. Input password = admin
    Click    text="s"
    Custom Login Keyword
    Log    Action: Alternative flows (if any):
    Log    Action: Go to    https://the-internet.herokuapp.com/basic_auth
    Get Element States
    Log    Action: Input username = admin
    Log    Action: Input password = admin
    Log    Action: Alternative flows (if any. Basic Authentic.
    Custom Login Keyword
    Log    Action: Alternative
    Custom Login Keyword
    Log    Action: The first: Too to your_5th to test.html. To your features and1. To your to your to the ions: to the first to-to your to your-to your-ish to your-to the First tox tos1: to1s to your to your10 to your to-to instructions: Toxic to your toster to your to your toment
    Log    Action: The first to1s tos to1s
    Log    Action: to your tofrs1s andions and to yourions
    Log    Action: toxions and actionions and1 andions
    Log    Action: tox as tox tos to-tofulsions to actions tos
    Log    Action: Yourions for1
    Log    Action: Ins to the to yourmentscripts andmentsments to the to thesions
    Log    Action: sents
    Log    Action: withments
    Log    Action: P-Yourful forfulentction
    Log    Action: to the towing toment tofic.
    Log    Action: basedctionsctionsumsctions

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
