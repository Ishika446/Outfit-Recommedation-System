const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

form.addEventListener("submit", async (e) => {

    e.preventDefault();

    const input = document.getElementById("question");
    const question = input.value;

    addMessage(question,"user");

    input.value = "";

    const loading = document.createElement("div");
    loading.className = "message bot";

    loading.innerHTML = `
        <div class="loading">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    chatBox.appendChild(loading);

    chatBox.scrollTop = chatBox.scrollHeight;

    try{

        const response = await fetch("/generate",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                question:question
            })
        });

        const data = await response.json();

        loading.remove();

        typeMessage(data.answer);

    }
    catch(error){

        loading.remove();

        addMessage(
            "Something went wrong.",
            "bot"
        );
    }

});

function addMessage(text,type){

    const div = document.createElement("div");

    div.className = `message ${type}`;

    div.innerHTML = text;

    chatBox.appendChild(div);

    chatBox.scrollTop = chatBox.scrollHeight;
}

function typeMessage(text){

    const div = document.createElement("div");

    div.className = "message bot";

    chatBox.appendChild(div);

    let i = 0;

    const timer = setInterval(()=>{

        div.innerHTML += text.charAt(i);

        i++;

        chatBox.scrollTop = chatBox.scrollHeight;

        if(i >= text.length){
            clearInterval(timer);
        }

    },20);

}