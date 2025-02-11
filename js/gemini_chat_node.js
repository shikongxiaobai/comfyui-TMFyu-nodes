import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.GeminiChat",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "GeminiChat") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function(message) {
				onExecuted?.apply(this, [message]);
				const node = this;
				const widget = this.widgets.find((w) => w.name === "text");
				if (widget && message.result) {
					const result = message.result[0];
					const div = document.createElement("div");
					div.classList.add("comfy-multiline-input");
					div.style.position = "absolute";
					div.style.left = "0px";
					div.style.width = "calc(100% - 5px)";
					div.style.top = "0px";
					div.style.height = "100%";
					div.style.pointerEvents = "none";

					const textarea = document.createElement("textarea");
					textarea.style.width = "100%";
					textarea.style.height = "100%";
					textarea.value = result;
					textarea.readOnly = true;
					textarea.style.color = "#9f9";
					textarea.style.fontSize = "1.2em";
					div.append(textarea);

					widget.inputEl.style.display = "none";
					widget.inputEl.parentNode.append(div);
				}
			};
		}
	},
});
