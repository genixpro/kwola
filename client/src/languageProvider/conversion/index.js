import ch from './raw/chenese.js';
import fr from './raw/fr.js';
import ital from './raw/ital.js';
import span from './raw/span.js';
import arab from './raw/arab.js';
import english from './raw/eng.js';

const sortObject = object => {
  const newObj = {};
  const keys = Object.keys(object).sort();
  keys.forEach(key => (newObj[key] = object[key]));
  return newObj;
};
export function getKeys(object) {
  object = sortObject(object);
  let variables = [];
  let text = '';
  let p = 0;
  Object.keys(object).forEach(key => {
    variables.push(object[key]);
    text += object[key] + '\n';
    p++;
  });
  // console.log('\n',JSON.stringify(object, null, 2),'\n', text,'\n', Object.keys(object).length, p);
  return {
    keys: Object.keys(object),
    variables,
  };
}
export function getValues(enMessages) {
  const { keys, variables } = getKeys(enMessages);
  const langs = [english, ch, fr, ital, span, arab];
  const langsNm = ['eng', 'ch', 'fr', 'ital', 'span', 'arab'];
  langs.forEach((lang, ii) => {
    const translatedDAta = lang.split('\n');
    const obj = {};
    keys.forEach((key, index) => {
      obj[key] = translatedDAta[index + 1];
    });
    console.log(
      '\n\n',
      langsNm[ii],
      translatedDAta.length,
      keys.length,
      '\n\n',
      JSON.stringify(obj, null, 2)
    );
  });
}
