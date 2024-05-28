elements = [];
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'convert') {
    // document.documentElement.style.filter = 'grayscale(100%)';
    // alert("heh");
  function addStyleString(str) {
      var node = document.createElement('style');
      node.innerHTML = str;
      document.body.appendChild(node);
  }
  
  addStyleString('.-sitemap-select-item-hover{outline:2px solid green!important}.-sitemap-select-item-hover,.-sitemap-select-item-hover *{background-color:rgba(0,213,0,.2)!important;background:rgba(0,213,0,.2)!important}.-sitemap-select-item-selected{outline:2px solid #c70000!important}.-sitemap-select-item-selected,.-sitemap-select-item-selected *{background-color:rgba(213,0,0,.2)!important;background:rgba(213,0,0,.2)!important}')
  // addStyleString('.-sitemap-select-item-hover2{outline:2px solid green!important}.-sitemap-select-item-hover,.-sitemap-select-item-hover *{background-color:rgba(0,213,0,.2)!important;background:rgba(0,213,0,.2)!important}.-sitemap-select-item-selected{outline:2px solid red!important}.-sitemap-select-item-selected,.-sitemap-select-item-selected *{background-color:rgba(213,0,0,.2)!important;background:rgba(213,0,0,.2)!important}')
  addStyleString('.-sitemap-select-item-hover2{outline:2px solid red!important}.-sitemap-select-item-hover,.-sitemap-select-item-hover *{background-color:rgba(255,0,0,.2)!important;background:rgba(255,0,0,.2)!important}.-sitemap-select-item-selected{outline:2px solid darkred!important}.-sitemap-select-item-selected,.-sitemap-select-item-selected *{background-color:rgba(139,0,0,.2)!important;background:rgba(139,0,0,.2)!important}')

  async function sendDetectionRequest(data) {
    const json = `{"data": "${data}"}`;
    const obj = JSON.parse(json);
    const response = await fetch('http://127.0.0.1:5000/detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: obj
    });

    if (!response.ok) {
        throw new Error('Network response was not ok');
    }

    const result = await response.json();
    return result;
  }

  function mouseover(event) {
      event.target.classList.add('-sitemap-select-item-hover')
  }
  function mouseout(event) {
      event.target.classList.remove('-sitemap-select-item-hover')
  }
  function clicking(event){
      event.preventDefault();
      element = document.elementFromPoint(event.clientX, event.clientY);
      elements.push(element);
      
      // alert(element.innerText);
      
      let xD = sendDetectionRequest('some plain text data');
      alert(xD);

      document.removeEventListener("mouseout", mouseout);
      document.removeEventListener("mouseover", mouseover);
      document.removeEventListener("click", clicking);
  }
  
  lol = document.addEventListener('mouseover', mouseover)
  kekw = document.addEventListener('mouseout', mouseout)
  document.addEventListener('click', clicking);

  sendResponse({ message: 'Website converted to black and white.' });
  } else if (request.action === 'reset') {
    elements.forEach((element) => {
      // if(element.style.visibility === "hidden")
      if(element.classList.contains("-sitemap-select-item-hover2"))
        element.classList.remove('-sitemap-select-item-hover2');
      else
        element.classList.add('-sitemap-select-item-hover2');
        // element.style.visibility = "hidden";
    })
    document.documentElement.style.filter = 'none';
    sendResponse({ message: 'Website reset to original colors.' });
  }
});